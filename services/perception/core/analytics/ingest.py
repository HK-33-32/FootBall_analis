"""Turn a perception run into the tables the ledger builder works on.

Input is whatever the GSR pipeline already writes: predictions.json (one row per
detection, pitch coordinates included), the attrs_*.json sidecar with per-track
name confidence, and job.json for fps and the offset into the source video.
Nothing here re-runs a model — the analytics stack is decoupled from perception
on purpose, so it can be iterated on in seconds against a finished job.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

PLAYER_ROLES = ("player", "goalkeeper")


@dataclass
class Run:
    """Everything one analysed clip contributes."""
    detections: pd.DataFrame     # role != ball
    ball: pd.DataFrame           # role == ball, one row per detection
    players: dict                # player_id -> registry entry
    track_to_player: dict        # track_id -> player_id
    fps: float
    start_s: float               # offset of the clip inside the source video
    frames: np.ndarray           # sorted frame indices that carry any detection
    meta: dict


def _text(value):
    """Attribute strings arrive as None, as NaN, or as the sentinel '100'.

    The pipeline writes missing jersey numbers and names through pandas, so they
    reach the JSON as bare NaN — which json.load happily returns as a float and
    which is truthy.  Left alone it produces a player called `right_nan` that
    swallows every unidentified track on that side.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "100":
        return None
    return text


def _load_attrs(job_dir: str) -> dict:
    """Name -> confidence, gathered from every segment of the run.

    A single-segment job writes its data at the job root; a segmented one puts
    each segment under seg000/, seg001/ and so on, and looking only at the root
    left every player in a long run with a confidence of zero.

    The sidecar is keyed by track id, but `merge_predictions` shifts track ids
    when it stitches segments together, so those ids no longer address anything
    in the merged file.  The confidence belongs to the *name* the roster match
    produced, so that is what it is keyed by here — correct for a segmented run
    and unchanged in meaning for a single one.
    """
    attrs: dict = {}
    roots = [job_dir]
    roots += sorted(os.path.join(job_dir, name) for name in os.listdir(job_dir)
                    if name.startswith("seg")
                    and os.path.isdir(os.path.join(job_dir, name)))
    for root in roots:
        split = os.path.join(root, "data", "SoccerNetGS", "test")
        if not os.path.isdir(split):
            continue
        for clip in sorted(os.listdir(split)):
            side = os.path.join(split, clip, "attrs_%s.json" % clip)
            if not os.path.isfile(side):
                continue
            with open(side, encoding="utf-8") as fh:
                for value in json.load(fh).values():
                    name = value.get("name")
                    score = float(value.get("confidence", 0.0))
                    if name and score > attrs.get(name, 0.0):
                        attrs[name] = score
    return attrs


def _pitch_xy(box):
    if not box:
        return np.nan, np.nan
    return float(box.get("x_bottom_middle", np.nan)), float(box.get("y_bottom_middle", np.nan))


def load_job(job_dir: str) -> Run:
    job_dir = os.path.abspath(job_dir)
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as fh:
        job = json.load(fh)
    pred_path = os.path.join(job_dir, "predictions.json")
    with open(pred_path, encoding="utf-8") as fh:
        predictions = json.load(fh)["predictions"]

    attrs = _load_attrs(job_dir)

    rows = []
    for p in predictions:
        attr = p.get("attributes") or {}
        x, y = _pitch_xy(p.get("bbox_pitch"))
        img = p.get("bbox_image") or {}
        rows.append((
            int(p["frame"]), int(p.get("track_id") or -1),
            _text(attr.get("role")) or "player", _text(attr.get("team")),
            _text(attr.get("jersey")), _text(attr.get("name")), x, y,
            float(img.get("x", np.nan)), float(img.get("y", np.nan)),
            float(img.get("w", np.nan)), float(img.get("h", np.nan)),
        ))
    df = pd.DataFrame(rows, columns=[
        "frame", "track_id", "role", "team", "jersey", "name",
        "x", "y", "img_x", "img_y", "img_w", "img_h"])

    ball = df[df.role == "ball"].copy().sort_values("frame").reset_index(drop=True)
    people = df[df.role.isin(PLAYER_ROLES + ("referee",))].copy()

    players, track_to_player = _build_registry(people, attrs)
    people["player_id"] = people.track_id.map(track_to_player)

    fps = float(job.get("fps") or 25)
    meta = {
        "job_id": job.get("id"),
        "video_name": job.get("video_name"),
        "fps": fps,
        "start_s": float(job.get("start") or 0.0),
        "end_s": float(job.get("end") or 0.0),
        "detector": job.get("detector"),
        "jersey_reader": job.get("jersey_reader"),
        "frames_total": int(df.frame.max()) if len(df) else 0,
        "frames_with_detections": int(df.frame.nunique()),
        "frames_with_ball": int(ball.frame.nunique()),
    }
    return Run(detections=people, ball=ball, players=players,
               track_to_player=track_to_player, fps=fps,
               start_s=meta["start_s"],
               frames=np.array(sorted(df.frame.unique())), meta=meta)


def _build_registry(people: pd.DataFrame, attrs: dict):
    """Collapse tracks into players.

    A track is a fragment, not a person: the same player picks up a new track id
    after every occlusion.  Where a shirt number was read, (team, number) is the
    identity and all fragments carrying it become one player.  Where it was not,
    the fragment stays its own anonymous player rather than being merged on a
    guess — an anonymous row in the output is honest, a wrong merge is not.
    """
    players: dict = {}
    track_to_player: dict = {}

    for track_id, group in people.groupby("track_id"):
        roles = Counter(r for r in group["role"] if _text(r))
        role = roles.most_common(1)[0][0] if roles else "player"
        teams = Counter(t for t in group["team"] if _text(t))
        team = teams.most_common(1)[0][0] if teams else None
        numbers = Counter(j for j in group["jersey"] if _text(j))
        jersey = numbers.most_common(1)[0][0] if numbers else None
        names = Counter(n for n in group["name"] if _text(n))
        name = names.most_common(1)[0][0] if names else None
        confidence = float(attrs.get(name, 0.0)) if name else 0.0

        if role == "referee":
            player_id = "referee_%d" % track_id
        elif jersey and team:
            player_id = "%s_%s" % (team, jersey)
        else:
            player_id = "%s_unknown_t%d" % (team or "none", track_id)

        entry = players.setdefault(player_id, {
            "player_id": player_id, "team": team, "jersey": jersey,
            "name": name, "role": role, "attribution_confidence": 0.0,
            "track_ids": [], "frames": 0,
        })
        entry["track_ids"].append(int(track_id))
        entry["frames"] += int(group.frame.nunique())
        # a player's attribution confidence is the best evidence any of its
        # fragments produced, weighted down when the number was never read
        entry["attribution_confidence"] = max(
            entry["attribution_confidence"],
            confidence if jersey else min(confidence, 0.2))
        if name and not entry["name"]:
            entry["name"] = name
        if entry["role"] != "goalkeeper" and role == "goalkeeper":
            entry["role"] = "goalkeeper"
        track_to_player[int(track_id)] = player_id

    return players, track_to_player


def attacking_directions(run: Run) -> dict:
    """Which way each team attacks, from where the goalkeepers stand.

    A keeper sits in front of the goal he defends, so the team whose keeper is
    at negative x attacks towards +x.  Falls back to the mean outfield position
    when no keeper was seen.
    """
    out = {}
    keepers = run.detections[run.detections.role == "goalkeeper"]
    for team in ("left", "right"):
        rows = keepers[keepers.team == team]
        if len(rows) >= 5:
            out[team] = bool(np.nanmedian(rows.x) < 0)
            continue
        rows = run.detections[(run.detections.team == team) &
                              (run.detections.role == "player")]
        out[team] = bool(np.nanmean(rows.x) < 0) if len(rows) else team == "left"
    if len(out) == 2 and out["left"] == out["right"]:
        out["right"] = not out["left"]
    return out
