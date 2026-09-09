"""Tracklet-level team affiliation and jersey number.

Two problems with deciding these per frame:

* **Teams.** The pipeline buckets players by the colour *word* the CLIP head
  predicts ("white" / "grey" / "charcoal" …), keeps the two most frequent words
  and drops everything else to ``team = -1``; those detections then fall back to
  "whichever half of the pitch the player stands in", so an attacking player
  gets the opponent's colour.
* **Numbers.** The head returns an ``argmax`` even when it is 20 % sure, and the
  tracklet vote counts those guesses like any other.

This module follows the SoccerNet game-state baseline (sn-gamestate):
appearance embeddings of a whole tracklet are clustered into two teams with
KMeans, and attributes are voted per tracklet — here weighted by the model's own
confidence and by how large (i.e. readable) the crop was.

It rewrites the ``team`` and ``jersey`` columns of ``refined_<clip>.txt`` and so
runs between IDATR and ``create_court_file``.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from fast_prep import shared_reid, thread_map
from roster import (Roster, assign_numbers, correct_teams,
                    match_clusters_to_teams)

COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "score", "role", "jersey",
           "color", "team"]
PLAYER_ROLES = ("Player", "Goalkeeper")

MAX_CROPS_PER_TRACK = 60      # largest crops of the tracklet
MAX_CROP_HEIGHT = 320         # close-ups carry no extra information for these heads
REID_CHUNK = 256
FRAME_BLOCK = 32              # frames decoded at once when cutting crops
TORSO_TOP, TORSO_BOTTOM = 0.10, 0.60   # kit region of the box
MIN_TRACK_CROPS = 3


# --------------------------------------------------------------------- crops
def _sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per tracklet keep the largest boxes: those carry a readable number."""
    parts = []
    for track_id, group in df.groupby("track_id"):
        parts.append(group.nlargest(MAX_CROPS_PER_TRACK, "h"))
    return pd.concat(parts) if parts else df.iloc[:0]


def _load_crops(clip_dir: str, rows: pd.DataFrame):
    """Returns crops, torso crops and the row index they belong to."""
    img_dir = os.path.join(clip_dir, "img1")
    crops, torsos, keep = [], [], []
    groups = list(rows.groupby("frame"))

    def read(item):
        return cv2.imread(os.path.join(img_dir, "%06d.jpg" % int(item[0])))

    # Decoding is most of the cost here and releases the GIL, so frames are read
    # a block at a time. A block, not the clip: a minute of 1080p is nine
    # gigabytes of decoded frames, and each is only needed long enough to be
    # cut. Cutting stays in frame order, so the output is unchanged.
    for start in range(0, len(groups), FRAME_BLOCK):
        block = groups[start:start + FRAME_BLOCK]
        for (frame_id, group), image in zip(block, thread_map(read, block)):
            if image is None:
                continue
            height, width = image.shape[:2]
            for idx, row in group.iterrows():
                x1, y1 = max(0, int(row.x)), max(0, int(row.y))
                x2 = min(width, int(row.x + row.w))
                y2 = min(height, int(row.y + row.h))
                if x2 - x1 < 4 or y2 - y1 < 8:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.shape[0] > MAX_CROP_HEIGHT:
                    scale = MAX_CROP_HEIGHT / crop.shape[0]
                    crop = cv2.resize(crop, (max(4, int(crop.shape[1] * scale)), MAX_CROP_HEIGHT),
                                      interpolation=cv2.INTER_AREA)
                box_h = y2 - y1
                ty1 = y1 + int(box_h * TORSO_TOP)
                ty2 = y1 + int(box_h * TORSO_BOTTOM)
                torso = image[ty1:max(ty1 + 2, ty2), x1:x2]
                # the slices above are views into the frame, which is about to
                # be dropped -- copy so the frame itself can be freed
                crops.append(np.ascontiguousarray(crop))
                torsos.append(np.ascontiguousarray(torso))
                keep.append(idx)
    return crops, torsos, keep


def _torso_histogram(torso: np.ndarray) -> np.ndarray:
    """Colour signature of the kit: Lab histogram with pitch green removed."""
    small = cv2.resize(torso, (24, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    grass = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) &
             (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40))
    mask = (~grass).astype(np.uint8)
    if mask.sum() < 20:
        mask = np.ones_like(mask)
    hist = []
    for channel, bins, rng in ((0, 8, [0, 256]), (1, 8, [0, 256]), (2, 8, [0, 256])):
        h = cv2.calcHist([lab], [channel], mask, [bins], rng).flatten()
        hist.append(h)
    hist = np.concatenate(hist)
    total = hist.sum()
    return hist / total if total else hist


# ----------------------------------------------------------------- attributes
def _number_lookup(get_number):
    """(digit1, digit2) -> jersey number, for the 11x11 head output."""
    table = np.zeros((11, 11), dtype=np.int32)
    for d1 in range(11):
        for d2 in range(11):
            table[d1, d2] = get_number(d1, d2)
    return table


def _jersey_votes(probs1, probs2, weights, number_table):
    """Vote over a whole tracklet -> ({number: score}, unknown mass).

    The head predicts the two digits separately, so instead of taking an argmax
    per digit we accumulate the full joint distribution: every crop contributes
    its probability mass to each candidate number, weighted by how readable the
    crop was.  A crop the model is unsure about therefore adds little, instead
    of casting a full vote for a number it barely saw.
    """
    joint = (probs1.unsqueeze(2) * probs2.unsqueeze(1)).numpy()   # (N, 11, 11)
    joint = joint * np.asarray(weights, dtype=np.float32)[:, None, None]
    mass = joint.sum(axis=0)                                       # (11, 11)

    votes: dict[int, float] = {}
    unknown = 0.0
    for d1 in range(11):
        for d2 in range(11):
            number = int(number_table[d1, d2])
            value = float(mass[d1, d2])
            if number == 100:
                unknown += value
            else:
                votes[number] = votes.get(number, 0.0) + value
    return votes, unknown


def _box_iou(boxes, first, second) -> float:
    """Median IoU of two tracklets over the frames they share."""
    a, b = boxes.get(first) or {}, boxes.get(second) or {}
    common = a.keys() & b.keys()
    if not common:
        return 0.0
    values = []
    for frame in common:
        ax, ay, aw, ah = a[frame]
        bx, by, bw, bh = b[frame]
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        values.append(inter / union if union > 0 else 0.0)
    values.sort()
    return values[len(values) // 2]


def link_tracklets(track_ids, features, spans, boxes, threshold=0.7, limit=400):
    """Group tracklets that are the same player -> {track_id: group_id}.

    The tracker leaves a player in pieces — 73 tracklets for 22 players in a
    minute — and deciding team and number on each piece separately is what makes
    one fragment disagree with the next.  OSNet is identity-oriented, which made
    it a weak team signal but makes it the right tool here.

    Two tracklets may only join if they are never on screen at the same time,
    unless their boxes coincide, which is one player detected twice.
    """
    if len(track_ids) < 2 or len(track_ids) > limit:
        return {t: t for t in track_ids}

    groups = {t: [t] for t in track_ids}
    vectors = {t: features[i] / (np.linalg.norm(features[i]) + 1e-9)
               for i, t in enumerate(track_ids)}

    def mergeable(head_a, head_b):
        for a in groups[head_a]:
            for b in groups[head_b]:
                span_a, span_b = spans.get(a), spans.get(b)
                if not span_a or not span_b:
                    continue
                if span_a[0] <= span_b[1] and span_b[0] <= span_a[1]                         and _box_iou(boxes, a, b) < 0.5:
                    return False
        return True

    while True:
        best, best_similarity = None, threshold
        heads = list(groups)
        for i, head_a in enumerate(heads):
            for head_b in heads[i + 1:]:
                similarity = float(vectors[head_a] @ vectors[head_b])
                if similarity > best_similarity and mergeable(head_a, head_b):
                    best, best_similarity = (head_a, head_b), similarity
        if best is None:
            break
        head_a, head_b = best
        groups[head_a].extend(groups.pop(head_b))
        merged = vectors[head_a] + vectors.pop(head_b)
        vectors[head_a] = merged / (np.linalg.norm(merged) + 1e-9)

    return {track: head for head, members in groups.items() for track in members}



def _cluster_teams(track_ids, reid_feats, hist_feats, prtreid_feats=None):
    """KMeans on tracklet appearance; the better-separated feature set wins."""
    if len(track_ids) < 2:
        return {track_ids[0]: 0} if track_ids else {}

    best_labels, best_score, best_name = None, -2.0, "-"
    for name, feats in (("reid", reid_feats), ("colour", hist_feats),
                        ("prtreid", prtreid_feats)):
        if feats is None or feats.shape[0] < 2:
            continue
        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(norm)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(norm, labels)
        # a football clip has two roughly equal teams: punish lopsided splits
        share = min(np.bincount(labels)) / len(labels)
        score = score * (1.0 if share >= 0.25 else 0.5)
        print("  кластеризация по %-6s: silhouette %.3f, баланс %.0f/%.0f" %
              (name, score, share * 100, (1 - share) * 100), flush=True)
        if score > best_score:
            best_labels, best_score, best_name = labels, score, name

    if best_labels is None:
        return {tid: 0 for tid in track_ids}
    print("  команды по признакам: %s" % best_name, flush=True)
    return {tid: int(label) for tid, label in zip(track_ids, best_labels)}


def _vlm_enabled(cfg) -> bool:
    return str(cfg.get("JERSEY_READER", "clip")).lower() in ("qwen-vl", "vlm")


def _vlm_pass(cfg, scores_by_track, crops_by_track, allowed=None):
    """Let a local VLM re-read the numbers and fold its answer into the scores.

    It is not a hard override: the VLM answer is added as a strong vote, so a
    number that the roster forbids or that the VLM only saw once still loses to
    solid CLIP evidence.
    """
    if not _vlm_enabled(cfg):
        return scores_by_track

    from jersey_vlm import QwenJerseyReader

    vlm_cfg = cfg.get("JERSEY_VLM") or {}
    model = vlm_cfg.get("MODEL_PATH")
    mmproj = vlm_cfg.get("MMPROJ_PATH")
    if not QwenJerseyReader.available(model, mmproj):
        print("[attrs] VLM не найден (%s), номера остаются от CLIP" % model, flush=True)
        return scores_by_track

    torch.cuda.empty_cache()          # free the GPU before the VLM takes ~9 GB
    started = time.perf_counter()
    reader = QwenJerseyReader(model, mmproj,
                              crops_per_track=int(vlm_cfg.get("CROPS_PER_TRACK", 24)),
                              min_votes=int(vlm_cfg.get("MIN_VOTES", 1)),
                              group_size=int(vlm_cfg.get("GROUP_SIZE", 4)))
    print("[attrs] VLM загружен за %.1f с" % (time.perf_counter() - started), flush=True)

    weight = float(vlm_cfg.get("VOTE_WEIGHT", 2.0))
    started = time.perf_counter()
    changed = 0
    max_tracks = int(vlm_cfg.get("MAX_TRACKS", 200))
    ordered = sorted(crops_by_track.items(), key=lambda kv: -len(kv[1]))[:max_tracks]
    for track_id, crops in ordered:
        if len(crops) < 3:
            continue          # a fragment that downstream filtering drops anyway
        tally, answers = reader.tally(crops, allowed)
        if not tally:
            continue
        scores = scores_by_track.setdefault(track_id, {})
        mass = sum(scores.values()) or 1.0
        before = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        for number, votes in tally.items():
            scores[number] = scores.get(number, 0.0) +                 weight * mass * (votes / max(1, len(answers)))
        after = max(scores.items(), key=lambda kv: kv[1])[0]
        if before != after:
            print("[attrs] трек %s: CLIP %s -> VLM %s %s" %
                  (track_id, before, after, answers), flush=True)
            changed += 1
    print("[attrs] VLM: %d треков из %d за %.1f с, изменено номеров: %d" %
          (len(ordered), len(crops_by_track), time.perf_counter() - started, changed),
          flush=True)
    del reader
    return scores_by_track


def refine_tracklet_attributes(cfg, splits, device=None):
    from yolox.utils.transforms import get_transforms
    from jersey_model.CLIPFinetune import CLIPFinetune
    from jersey_batch import get_number

    number_table = _number_lookup(get_number)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_sum = float(cfg.get("JERSEY_MIN_VOTE", 1.0))
    unknown_ratio = float(cfg.get("JERSEY_UNKNOWN_RATIO", 0.15))

    roster = Roster.load(cfg.get("ROSTER"))
    if roster is not None:
        print("[attrs] заявка: %s — %s" % (roster.name, roster.describe()), flush=True)

    reid = shared_reid("osnet_x1_0", cfg["REID"]["MODEL_PATH"], device)
    transforms = get_transforms(image_size=(256, 128))
    jersey_model = CLIPFinetune.from_checkpoint(cfg["CLIP"]["MODEL_PATH"])
    jersey_model = jersey_model.to(device).eval()

    for split in splits:
        split_dir = os.path.join(cfg["DATA_DIR"], split)
        for clip in sorted(os.listdir(split_dir)):
            clip_dir = os.path.join(split_dir, clip)
            path = os.path.join(clip_dir, "refined_%s.txt" % clip)
            if not os.path.isfile(path):
                continue
            started = time.perf_counter()
            df = pd.read_csv(path, header=None, names=COLUMNS)
            players = df[df.role.isin(PLAYER_ROLES)]
            if players.empty:
                continue

            rows = _sample_rows(players)
            crops, torsos, index = _load_crops(clip_dir, rows)
            if len(crops) < MIN_TRACK_CROPS:
                continue

            with torch.no_grad():
                # build the batch per chunk: on a ten-minute clip there are tens
                # of thousands of crops and one big tensor would not fit
                embeddings = []
                for start in range(0, len(crops), REID_CHUNK):
                    part = torch.stack(thread_map(
                        lambda c: transforms(image=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))["image"],
                        crops[start:start + REID_CHUNK])).to(device)
                    feats = reid(part)
                    embeddings.append(F.normalize(feats, p=2, dim=1).cpu())
                    del part
                embeddings = torch.cat(embeddings).numpy()

                probs1, probs2 = [], []
                for start in range(0, len(crops), 96):
                    batch = torch.stack(thread_map(
                        lambda c: jersey_model.preprocess(
                            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))),
                        crops[start:start + 96])).to(device)
                    out = jersey_model(batch)
                    probs1.append(F.softmax(out["digit1_logits"].float(), 1).cpu())
                    probs2.append(F.softmax(out["digit2_logits"].float(), 1).cpu())
                probs1 = torch.cat(probs1)
                probs2 = torch.cat(probs2)

            histograms = np.stack(thread_map(_torso_histogram, torsos))

            prtreid = None
            if cfg.get("USE_PRTREID"):
                # PRTReID is trained on football, so its embedding is a second,
                # independent view of the kit; the silhouette selector below
                # decides whether it beats plain torso colour on this match.
                from prtreid_features import PRTReIDFeatures
                if PRTReIDFeatures.available():
                    extractor = PRTReIDFeatures(device=str(device))
                    prtreid, prt_roles = extractor(crops)
                    del extractor
                    torch.cuda.empty_cache()
                    print("[attrs] PRTReID: роли %s" %
                          {r: prt_roles.count(r) for r in set(prt_roles)}, flush=True)
                else:
                    print("[attrs] веса PRTReID не найдены, признак пропущен", flush=True)
            sub = rows.loc[index]
            # a number is only readable on a reasonably tall crop
            readability = np.clip(sub.h.to_numpy(dtype=float) / 90.0, 0.2, 1.0)

            print("[attrs] %s: %d треклетов, %d кропов" %
                  (clip, sub.track_id.nunique(), len(crops)), flush=True)

            track_ids, reid_means, hist_means, prt_means = [], [], [], []
            scores_by_track, unknown_by_track, crops_by_track = {}, {}, {}
            spans, roles_by_track, boxes_by_track = {}, {}, {}
            for track_id, group in sub.groupby("track_id"):
                mask = sub.track_id.to_numpy() == track_id
                if mask.sum() < MIN_TRACK_CROPS:
                    continue
                scores, unknown = _jersey_votes(
                    probs1[mask], probs2[mask], readability[mask], number_table)
                scores_by_track[track_id] = scores
                unknown_by_track[track_id] = unknown
                order = np.argsort(-sub.h.to_numpy(dtype=float)[mask])
                picked = np.where(mask)[0][order]
                crops_by_track[track_id] = [crops[i] for i in picked[:8]]

                track_rows = df.loc[df.track_id == track_id]
                frames = track_rows["frame"].astype(int)
                spans[track_id] = (int(frames.min()), int(frames.max()))
                boxes_by_track[track_id] = {
                    int(f): (float(x), float(y), float(w), float(h))
                    for f, x, y, w, h in zip(frames, track_rows.x, track_rows.y,
                                             track_rows.w, track_rows.h)}

                role = group.role.mode().iat[0] if not group.role.mode().empty else "Player"
                roles_by_track[track_id] = role
                if role == "Player":
                    track_ids.append(track_id)
                    reid_means.append(embeddings[mask].mean(axis=0))
                    hist_means.append(histograms[mask].mean(axis=0))
                    if prtreid is not None:
                        prt_means.append(prtreid[mask].mean(axis=0))

            teams = _cluster_teams(track_ids,
                                   np.stack(reid_means) if reid_means else None,
                                   np.stack(hist_means) if hist_means else None,
                                   np.stack(prt_means) if prt_means else None)

            # Fragments of one player are decided together: pooled evidence is
            # far steadier than a per-fragment guess, and a group can only have
            # one team and one number.
            group_of = {t: t for t in scores_by_track}
            if cfg.get("LINK_TRACKLETS", True) and reid_means:
                identity = {t: embeddings[sub.track_id.to_numpy() == t].mean(axis=0)
                            for t in scores_by_track}
                keys = list(identity)
                group_of = link_tracklets(
                    keys, np.stack([identity[t] for t in keys]), spans, boxes_by_track,
                    threshold=float(cfg.get("LINK_THRESHOLD", 0.7)))
                sizes = {}
                for track_id, head in group_of.items():
                    sizes.setdefault(head, []).append(track_id)
                merged = sum(1 for m in sizes.values() if len(m) > 1)
                print("  треков %d -> групп %d (объединено %d)"
                      % (len(group_of), len(sizes), merged), flush=True)

                # pool the evidence of every fragment onto its group
                pooled_scores, pooled_unknown, pooled_span = {}, {}, {}
                votes = {}
                for track_id, head in group_of.items():
                    target = pooled_scores.setdefault(head, {})
                    for number, value in scores_by_track[track_id].items():
                        target[number] = target.get(number, 0.0) + value
                    pooled_unknown[head] = pooled_unknown.get(head, 0.0) +                         unknown_by_track.get(track_id, 0.0)
                    span = spans.get(track_id)
                    if span:
                        current = pooled_span.get(head)
                        pooled_span[head] = span if current is None else (
                            min(current[0], span[0]), max(current[1], span[1]))
                    if track_id in teams:
                        weight = int((sub.track_id.to_numpy() == track_id).sum())
                        tally = votes.setdefault(head, {})
                        tally[teams[track_id]] = tally.get(teams[track_id], 0) + weight

                # one team per player, decided by the largest fragments
                fixed = 0
                for head, tally in votes.items():
                    winner = max(tally, key=tally.get)
                    for track_id in sizes[head]:
                        if track_id in teams and teams[track_id] != winner:
                            teams[track_id] = winner
                            fixed += 1
                if fixed:
                    print("  команда выровнена по группе у %d треков" % fixed, flush=True)

                # Pooling the number evidence across fragments sounds right but
                # measures worse: summing a confident "10" with fragments that
                # read something else blurs the distribution, and readings that
                # were plainly correct stop passing the margin test.  Teams are
                # decided per group (that demonstrably fixes errors), numbers
                # stay per fragment.
                if cfg.get("POOL_NUMBERS", False):
                    scores_by_track, unknown_by_track = pooled_scores, pooled_unknown
                    spans = pooled_span
                    crops_by_track = {head: crops_by_track[head]
                                      for head in pooled_scores if head in crops_by_track}
                    roles_by_track = {head: roles_by_track.get(head, "Player")
                                      for head in pooled_scores}
                    teams = {head: teams[head] for head in pooled_scores if head in teams}
                else:
                    group_of = {t: t for t in scores_by_track}

            if _vlm_enabled(cfg):
                # the VLM wants ~9 GB, so park the torch models on the CPU first
                jersey_model.to("cpu")
                reid.model.to("cpu")
                torch.cuda.empty_cache()
                allowed = None
                if roster is not None:
                    allowed = roster.numbers(0) | roster.numbers(1)
                scores_by_track = _vlm_pass(cfg, scores_by_track, crops_by_track,
                                            allowed)
                jersey_model.to(device)
                reid.model.to(device)

            team_by_cluster = match_clusters_to_teams(scores_by_track, teams, roster)
            if roster is not None and team_by_cluster:
                for cluster, index in sorted(team_by_cluster.items()):
                    print("  кластер %d -> %s" % (cluster, roster.teams[index]["short"]
                                                  or roster.teams[index]["team"]), flush=True)
                fixed = correct_teams(scores_by_track, teams, roster, team_by_cluster)
                if fixed:
                    teams.update(fixed)
                    print("  команда исправлена по номеру у %d треков" % len(fixed),
                          flush=True)

            jersey_by_track, jersey_conf = assign_numbers(
                scores_by_track, unknown_by_track, spans, teams, roster,
                team_by_cluster, roles=roles_by_track, min_score=min_sum,
                unknown_ratio=unknown_ratio,
                margin=float(cfg.get("JERSEY_ROSTER_MARGIN", 0.6)),
                boxes=boxes_by_track, debug=True)

            for track_id, head in group_of.items():
                if head in jersey_by_track:
                    df.loc[df.track_id == track_id, "jersey"] = jersey_by_track[head]
            df["team"] = -1
            for track_id, head in group_of.items():
                if head in teams:
                    df.loc[df.track_id == track_id, "team"] = teams[head]

            df["frame"] = df["frame"].astype(int)
            df.to_csv(path, index=False, header=False)

            if roster is not None and team_by_cluster:
                attrs = {}
                for track_id, head in group_of.items():
                    number = jersey_by_track.get(head)
                    index = team_by_cluster.get(teams.get(head))
                    name = roster.name_of(index, number) if index is not None else None
                    attrs[str(track_id)] = {
                        "name": name,
                        "confidence": round(float(jersey_conf.get(head, 0.0)), 4),
                    }
                names = {k: v["name"] for k, v in attrs.items() if v["name"]}
                import json as _json
                (open(os.path.join(clip_dir, "attrs_%s.json" % clip), "w",
                      encoding="utf-8")).write(_json.dumps(attrs, ensure_ascii=False))
                print("[attrs] фамилий по заявке: %d" % len(names), flush=True)

            summary = ", ".join("#%s:%s" % (t, jersey_by_track.get(t, "-"))
                                for t in sorted(jersey_by_track))
            print("[attrs] %s: %.1f с | номера %s" %
                  (clip, time.perf_counter() - started, summary), flush=True)
