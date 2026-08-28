"""Post-perception game-state refinement.

The perception backend emits per-detection identity attributes (role, team,
jersey) that are decided frame by frame. Frame-local decisions are the dominant
GS-HOTA failure mode: the similarity used by the official evaluator is zero
whenever role, team or jersey disagree, so one collapsed team classifier
destroys the detection accuracy that the geometry already earned.

This module refines those attributes after perception, without retraining:

* teams are decided once per tracklet from torso appearance and mapped to the
  ``left``/``right`` pitch convention with goalkeeper and offside-line votes;
* tracklets that are the same player either side of a gap are chained, using
  pitch geometry alone or re-identification embeddings when the backend kept
  them;
* jersey numbers are then decided once per chained identity, weighting each
  tracklet's read by how large its crops were.

Everything here is unsupervised with respect to the benchmark: no ground-truth
annotation is read at any point.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

Detection = dict[str, Any]
FrameLoader = Callable[[int], "np.ndarray | None"]

PLAYER_ROLES = ("player", "goalkeeper")
# roles this module may move a tracklet between; the ball is never one of them
TRACKED_ROLES = ("player", "goalkeeper", "referee")


@dataclass(frozen=True)
class RefinementConfig:
    """Tunables for :func:`refine_predictions`."""

    frame_stride: int = 2
    min_box_height: float = 20.0
    torso_top: float = 0.12
    torso_bottom: float = 0.45
    torso_left: float = 0.25
    torso_right: float = 0.75
    min_torso_pixels: int = 12
    grass_hue: tuple[int, int] = (30, 95)
    grass_min_saturation: int = 60
    min_value: int = 25
    lightness_weight: float = 0.0
    min_separation_ratio: float = 3.0
    max_kit_mass_share: float = 0.62
    min_track_samples: int = 2
    goalkeeper_min_abs_x: float = 25.0
    touchline_abs_y: float = 32.0
    referee_max_speed_m_s: float = 7.0
    reconcile_roles: bool = True
    offside_rank: int = 2
    jersey_min_votes: int = 2
    jersey_min_share: float = 0.34
    identity_read_weight: float = 5.0
    identity_read_min_consensus: float = 0.34
    enforce_number_exclusivity: bool = True
    jersey_fill: bool = False
    jersey_propagate: bool = True
    link_tracklets: bool = False
    link_max_gap_frames: int = 400
    link_max_distance_m: float = 5.0
    reid_min_cosine: float = 0.85
    reid_max_distance_m: float = 15.0


@dataclass(frozen=True)
class JerseyTally:
    """What an identity-level reader answered, including its refusals.

    ``questions`` counts every question asked, so the abstentions are part of
    the record: a reader that answered "no number visible" to five of six
    groups of crops said something important, and dividing only by the answers
    it did give would throw it away.
    """

    votes: Counter
    questions: int

    @property
    def consensus(self) -> float:
        if not self.votes or self.questions <= 0:
            return 0.0
        return self.votes.most_common(1)[0][1] / self.questions


@dataclass
class TrackSummary:
    """Everything the refinement decides per tracklet."""

    track_id: int
    role: str
    samples: int = 0
    detections: int = 0
    descriptor: np.ndarray | None = None
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    team: str | None = None
    team_cluster: int | None = None
    team_margin: float = 0.0
    jersey: str | None = None
    jersey_votes: Counter = field(default_factory=Counter)
    start_frame: int = 0
    end_frame: int = 0
    first_point: tuple[float, float] = (0.0, 0.0)
    last_point: tuple[float, float] = (0.0, 0.0)
    median_height: float = 0.0


def torso_descriptor(
    frame: np.ndarray, bbox: dict[str, float], config: RefinementConfig
) -> np.ndarray | None:
    """Mean CIELAB colour of the non-grass torso pixels of one detection."""
    import cv2

    height, width = frame.shape[:2]
    x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
    if h < config.min_box_height:
        return None
    x0 = max(0, int(round(x + config.torso_left * w)))
    x1 = min(width, int(round(x + config.torso_right * w)))
    y0 = max(0, int(round(y + config.torso_top * h)))
    y1 = min(height, int(round(y + config.torso_bottom * h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    hue, saturation, value = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    low, high = config.grass_hue
    grass = (hue >= low) & (hue <= high) & (saturation > config.grass_min_saturation)
    keep = (~grass) & (value > config.min_value)
    if int(keep.sum()) < config.min_torso_pixels:
        return None
    return lab[keep].mean(axis=0)


def summarise_tracks(
    predictions: Sequence[Detection], frames: FrameLoader, config: RefinementConfig
) -> dict[int, TrackSummary]:
    """Collect per-tracklet appearance, position and jersey evidence."""
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    roles: dict[int, Counter] = defaultdict(Counter)
    pitch_x: dict[int, list[float]] = defaultdict(list)
    pitch_y: dict[int, list[float]] = defaultdict(list)
    jersey_votes: dict[int, Counter] = defaultdict(Counter)
    heights: dict[int, list[float]] = defaultdict(list)
    trajectory: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    counts: Counter = Counter()
    for detection in predictions:
        attributes = detection.get("attributes") or {}
        if attributes.get("role") not in TRACKED_ROLES:
            continue
        track_id = detection["track_id"]
        counts[track_id] += 1
        roles[track_id][attributes["role"]] += 1
        by_frame[int(detection["frame"])].append(detection)
        heights[track_id].append(float(detection["bbox_image"]["h"]))
        pitch = detection.get("bbox_pitch") or {}
        if "x_bottom_middle" in pitch:
            pitch_x[track_id].append(float(pitch["x_bottom_middle"]))
            pitch_y[track_id].append(float(pitch["y_bottom_middle"]))
            trajectory[track_id].append(
                (
                    int(detection["frame"]),
                    float(pitch["x_bottom_middle"]),
                    float(pitch["y_bottom_middle"]),
                )
            )
        jersey = attributes.get("jersey")
        if jersey not in (None, ""):
            jersey_votes[track_id][str(jersey)] += 1

    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_number in sorted(by_frame):
        if config.frame_stride > 1 and frame_number % config.frame_stride:
            continue
        image = frames(frame_number)
        if image is None:
            continue
        for detection in by_frame[frame_number]:
            descriptor = torso_descriptor(image, detection["bbox_image"], config)
            if descriptor is not None:
                samples[detection["track_id"]].append(descriptor)

    summaries: dict[int, TrackSummary] = {}
    for track_id, total in counts.items():
        observed = samples.get(track_id, [])
        path = sorted(trajectory[track_id])
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            role=roles[track_id].most_common(1)[0][0],
            samples=len(observed),
            detections=total,
            descriptor=np.median(np.asarray(observed), axis=0) if observed else None,
            pitch_x=float(np.median(pitch_x[track_id])) if pitch_x[track_id] else 0.0,
            pitch_y=float(np.median(pitch_y[track_id])) if pitch_y[track_id] else 0.0,
            jersey_votes=jersey_votes[track_id],
            start_frame=path[0][0] if path else 0,
            end_frame=path[-1][0] if path else 0,
            first_point=(path[0][1], path[0][2]) if path else (0.0, 0.0),
            last_point=(path[-1][1], path[-1][2]) if path else (0.0, 0.0),
            median_height=float(np.median(heights[track_id])) if heights[track_id] else 0.0,
        )
    return summaries


def _weighted_two_means(
    descriptors: np.ndarray, weights: np.ndarray, iterations: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic weighted 2-means seeded with the farthest descriptor pair."""
    distances = np.linalg.norm(descriptors[:, None, :] - descriptors[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    centres = descriptors[[first, second]].astype(float).copy()
    labels = np.full(len(descriptors), -1, dtype=int)
    for _ in range(iterations):
        assigned = np.argmin(
            np.linalg.norm(descriptors[:, None, :] - centres[None, :, :], axis=2), axis=1
        )
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for cluster in (0, 1):
            mask = labels == cluster
            if mask.any():
                centres[cluster] = np.average(descriptors[mask], axis=0, weights=weights[mask])
    return labels, centres


def _separation_ratio(
    descriptors: np.ndarray, weights: np.ndarray, labels: np.ndarray, centres: np.ndarray
) -> float:
    """How far apart the two kit centres are, in units of within-kit scatter."""
    residuals = np.linalg.norm(descriptors - centres[labels], axis=1)
    scatter = float(np.average(residuals, weights=weights))
    gap = float(np.linalg.norm(centres[0] - centres[1]))
    return gap / scatter if scatter > 1e-9 else float("inf")


def _mass_share(weights: np.ndarray, labels: np.ndarray) -> float:
    """Share of the observed detections that the larger of the two kits holds."""
    total = float(weights.sum())
    if total <= 0:
        return 1.0
    first = float(weights[labels == 0].sum())
    return max(first, total - first) / total


def _cluster_kits(
    descriptors: np.ndarray, weights: np.ndarray, config: RefinementConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Split tracklets into two kits by torso colour.

    Kit identity normally lives in chroma, while lightness mostly encodes
    whether a player is standing in sun or in shadow -- and lightness has by
    far the widest spread of the three CIELAB channels, so at full weight it
    splits each team by illumination instead of separating the two teams.
    ``lightness_weight`` therefore defaults to zero.

    One matchup breaks that: two kits that differ mainly in lightness, such as
    white stripes against navy. There the chroma-only split has nothing to work
    with, and it shows -- the two centres end up barely further apart than the
    scatter within each. When that happens the clustering is retried with
    lightness at full weight, which is the only evidence left.

    Separation alone does not always catch it. A chroma split can look
    comfortably separated and still be a split of something other than the two
    teams, and then it gives itself away by its size: both teams are on the
    pitch throughout, so a split that puts three quarters of the observed mass
    on one side is not a split by kit. Measured over four SoccerNet sequences
    the correct chroma split never sent more than 0.585 of the mass one way,
    while a broken one sent 0.761 -- so a lopsided split is retried too, and
    the retry is kept only if it is genuinely more even.
    """

    def split(lightness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        scale = np.array([lightness, 1.0, 1.0])
        scaled = descriptors * scale
        labels, centres = _weighted_two_means(scaled, weights)
        return labels, centres, scale, _separation_ratio(scaled, weights, labels, centres)

    result = split(config.lightness_weight)
    if config.lightness_weight >= 1.0:
        return result
    if result[3] < config.min_separation_ratio:
        return split(1.0)
    if _mass_share(weights, result[0]) > config.max_kit_mass_share:
        alternative = split(1.0)
        if _mass_share(weights, alternative[0]) < _mass_share(weights, result[0]):
            return alternative
    return result


def demote_surplus_referees(
    summaries: dict[int, TrackSummary], config: RefinementConfig
) -> list[int]:
    """Only one referee runs inside the field of play. Everyone else is a player.

    A match has one referee on the pitch; the assistants stay beyond the
    touchline, where no player ever goes, so a tracklet whose median position
    is outside ``touchline_abs_y`` is an official and is left alone. Inside the
    field, several simultaneous "referees" are a contradiction, and colour
    cannot always settle it -- on footage where the officials wear black and a
    team wears white, the two are indistinguishable once illumination is
    factored out.

    Movement settles it instead. The referee is one person, so his tracklets
    form a chain that is continuous in time and reachable at a jogging pace.
    The heaviest such chain is kept and every other in-field candidate is
    demoted. Measured on a clip where the backend called three yellow-shirted
    players referees: the genuine fragments chained at 2.9 and 4.8 m/s, while
    joining the impostor to the chain would have required 15.0 m/s.
    """
    officials = [summary for summary in summaries.values() if summary.role == "referee"]
    candidates = sorted(
        (s for s in officials if abs(s.pitch_y) < config.touchline_abs_y),
        key=lambda summary: (summary.start_frame, summary.track_id),
    )
    if len(candidates) < 2:
        return []

    count = len(candidates)
    best = [summary.detections for summary in candidates]
    previous = [-1] * count
    for later in range(count):
        for earlier in range(later):
            head, tail = candidates[earlier], candidates[later]
            gap = (tail.start_frame - head.end_frame) / 25.0
            if gap <= 0:
                continue
            distance = float(
                np.linalg.norm(np.asarray(head.last_point) - np.asarray(tail.first_point))
            )
            if distance > config.referee_max_speed_m_s * gap:
                continue
            if best[earlier] + tail.detections > best[later]:
                best[later] = best[earlier] + tail.detections
                previous[later] = earlier

    end = int(np.argmax(best))
    chain = set()
    while end != -1:
        chain.add(candidates[end].track_id)
        end = previous[end]

    demoted = []
    for summary in candidates:
        if summary.track_id not in chain:
            summary.role = "player"
            demoted.append(summary.track_id)
    return demoted


def _opposite(side: str) -> str:
    return "left" if side == "right" else "right"


def _side_of(pitch_x: float, config: RefinementConfig) -> str | None:
    if abs(pitch_x) < config.goalkeeper_min_abs_x:
        return None
    return "right" if pitch_x > 0 else "left"


def _goalkeeper_side(summaries: dict[int, TrackSummary], config: RefinementConfig) -> str | None:
    votes: Counter = Counter()
    for summary in summaries.values():
        if summary.role != "goalkeeper":
            continue
        side = _side_of(summary.pitch_x, config)
        if side is not None:
            votes[side] += summary.detections
    return votes.most_common(1)[0][0] if votes else None


def _offside_line_vote(
    summaries: dict[int, TrackSummary],
    predictions: Sequence[Detection],
    goalkeeper_side: str | None,
    config: RefinementConfig,
) -> tuple[int | None, dict[str, int]]:
    """Vote on which kit cluster defends the goalkeeper's goal.

    The player holding the offside line -- the ``offside_rank``-th outfield
    player counted from the defended goal line -- belongs to the defending team
    in the large majority of open-play frames.
    """
    if goalkeeper_side is None:
        return None, {}
    sign = 1.0 if goalkeeper_side == "right" else -1.0
    by_frame: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for detection in predictions:
        attributes = detection.get("attributes") or {}
        if attributes.get("role") != "player":
            continue
        summary = summaries.get(detection["track_id"])
        if summary is None or summary.team_cluster is None:
            continue
        pitch = detection.get("bbox_pitch") or {}
        if "x_bottom_middle" not in pitch:
            continue
        by_frame[int(detection["frame"])].append(
            (sign * float(pitch["x_bottom_middle"]), summary.team_cluster)
        )
    votes: Counter = Counter()
    for players in by_frame.values():
        if len(players) < config.offside_rank:
            continue
        players.sort(key=lambda item: -item[0])
        votes[players[config.offside_rank - 1][1]] += 1
    if not votes:
        return None, {}
    winner, _ = votes.most_common(1)[0]
    return int(winner), {str(cluster): count for cluster, count in votes.items()}


def _mean_x_sides(
    outfield: list[TrackSummary], labels: np.ndarray, weights: np.ndarray
) -> dict[int, str]:
    means: dict[int, float] = {}
    for cluster in (0, 1):
        mask = labels == cluster
        if mask.any():
            values = [summary.pitch_x for summary, keep in zip(outfield, mask, strict=True) if keep]
            means[cluster] = float(np.average(values, weights=weights[mask]))
        else:
            means[cluster] = 0.0
    right = max(means, key=lambda cluster: means[cluster])
    return {right: "right", 1 - right: "left"}


def demote_stranded_goalkeepers(
    summaries: dict[int, TrackSummary], config: RefinementConfig
) -> list[int]:
    """A goalkeeper standing at the halfway line is not a goalkeeper.

    The keeper label is worth challenging on geometry alone: the backend
    already decides a keeper's team from which goal he stands in front of, so a
    tracklet it calls a keeper while he spends the clip in midfield contradicts
    its own reasoning. Demoting him also puts his kit back into the team
    clustering, where it belongs.
    """
    demoted = []
    for summary in summaries.values():
        if summary.role != "goalkeeper":
            continue
        if abs(summary.pitch_x) >= config.goalkeeper_min_abs_x:
            continue
        summary.role = "player"
        demoted.append(summary.track_id)
    return demoted


def assign_teams(
    summaries: dict[int, TrackSummary],
    predictions: Sequence[Detection],
    config: RefinementConfig,
) -> dict[str, Any]:
    """Cluster outfield tracklets by kit colour and map the clusters to pitch sides."""
    demoted: list[int] = []
    surplus: list[int] = []
    if config.reconcile_roles:
        demoted = demote_stranded_goalkeepers(summaries, config)
        surplus = demote_surplus_referees(summaries, config)
    outfield = [
        summary
        for summary in summaries.values()
        if summary.role == "player"
        and summary.descriptor is not None
        and summary.samples >= config.min_track_samples
    ]
    report: dict[str, Any] = {
        "clustered_tracks": len(outfield),
        "unclustered_tracks": len(summaries) - len(outfield),
    }
    if len(outfield) < 2:
        report["status"] = "insufficient-evidence"
        return report

    descriptors = np.asarray([summary.descriptor for summary in outfield], dtype=float)
    weights = np.asarray([max(summary.samples, 1) for summary in outfield], dtype=float)
    labels, centres, scale, separation_ratio = _cluster_kits(descriptors, weights, config)
    # Short tracklets do not get a say in where the centres are, but they are
    # still assigned to the nearer one -- leaving them on the backend's own
    # team label would reintroduce exactly the error being corrected here.
    assigned: list[TrackSummary] = []
    for summary in summaries.values():
        if summary.role != "player" or summary.descriptor is None:
            continue
        distances = np.linalg.norm(centres - summary.descriptor * scale, axis=1)
        label = int(np.argmin(distances))
        summary.team_cluster = label
        summary.team_margin = float(distances[1 - label] - distances[label])
        assigned.append(summary)
    report["assigned_tracks"] = len(assigned)

    scatter = float(
        np.average(
            np.linalg.norm(descriptors * scale - centres[labels], axis=1),
            weights=weights,
        )
    )

    goalkeeper_side = _goalkeeper_side(summaries, config)
    anchor, offside_votes = _offside_line_vote(summaries, predictions, goalkeeper_side, config)
    fallback = _mean_x_sides(outfield, labels, weights)
    if anchor is None or goalkeeper_side is None:
        sides, rule = fallback, "mean-pitch-x"
    else:
        sides = {anchor: goalkeeper_side, 1 - anchor: _opposite(goalkeeper_side)}
        rule = "offside-line-vote"
    for summary in assigned:
        summary.team = sides[summary.team_cluster]
    for summary in summaries.values():
        if summary.role == "goalkeeper":
            side = _side_of(summary.pitch_x, config)
            if side is not None:
                summary.team = side

    report.update(
        {
            "status": "ok",
            "cluster_separation": round(float(np.linalg.norm(centres[0] - centres[1])), 3),
            "lightness_weight": float(scale[0]),
            "cluster_separation_ratio": round(separation_ratio, 3),
            "min_separation_ratio": config.min_separation_ratio,
            "kit_mass_share": round(_mass_share(weights, labels), 3),
            "max_kit_mass_share": config.max_kit_mass_share,
            "cluster_scatter": round(scatter, 3),
            "side_rule": rule,
            "goalkeeper_side": goalkeeper_side,
            "offside_votes": offside_votes,
            "fallback_sides": {str(key): value for key, value in fallback.items()},
            "resolved_sides": {str(key): value for key, value in sides.items()},
            "sides_agree": fallback == sides,
            "team_sizes": dict(Counter(summary.team for summary in assigned)),
            "goalkeepers_demoted": demoted,
            "surplus_referees_demoted": surplus,
        }
    )
    return report


def link_tracklets(
    summaries: dict[int, TrackSummary],
    config: RefinementConfig,
    embeddings: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Chain tracklets that are the same player seen before and after a gap.

    A tracklet may hand over to at most one successor, so the candidate pairs
    form a rectangular assignment problem over (tracklet end, tracklet start).
    A pair is admissible when it shares a team and role, does not overlap in
    time, and the two endpoints are close enough on the pitch.

    Without re-identification embeddings the pitch distance is the only
    appearance-free evidence available, so the gate has to stay tight
    (``link_max_distance_m``): a link is only proposed when the player has
    barely moved across the gap. When embeddings are supplied the gate widens
    to ``reid_max_distance_m`` and the decision is driven by embedding cosine
    similarity instead, which is what makes linking across a real run possible.
    """
    tracks = sorted(summaries.values(), key=lambda summary: (summary.start_frame, summary.track_id))
    index = {summary.track_id: position for position, summary in enumerate(tracks)}
    size = len(tracks)
    use_reid = bool(embeddings)
    max_distance = config.reid_max_distance_m if use_reid else config.link_max_distance_m
    blocked = 1e3
    cost = np.full((size, size), blocked, dtype=float)
    for earlier in tracks:
        for later in tracks:
            if earlier.track_id == later.track_id:
                continue
            gap = later.start_frame - earlier.end_frame
            if gap <= 0 or gap > config.link_max_gap_frames:
                continue
            if earlier.team != later.team or earlier.role != later.role:
                continue
            distance = float(
                np.linalg.norm(np.asarray(earlier.last_point) - np.asarray(later.first_point))
            )
            if distance > max_distance:
                continue
            if use_reid:
                left = embeddings.get(earlier.track_id)
                right = embeddings.get(later.track_id)
                if left is None or right is None:
                    continue
                similarity = float(np.dot(left, right))
                if similarity < config.reid_min_cosine:
                    continue
                cost[index[earlier.track_id], index[later.track_id]] = (
                    1.0 - similarity
                ) + 0.001 * distance
            else:
                cost[index[earlier.track_id], index[later.track_id]] = distance

    successor: dict[int, int] = {}
    if size:
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns, strict=True):
            if cost[row, column] < blocked:
                successor[tracks[row].track_id] = tracks[column].track_id

    parent: dict[int, int] = {summary.track_id: summary.track_id for summary in tracks}

    def root(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    for earlier_id, later_id in successor.items():
        parent[root(later_id)] = root(earlier_id)
    remap = {summary.track_id: root(summary.track_id) for summary in tracks}
    chains = len({value for value in remap.values()})
    return remap, {
        "links": len(successor),
        "tracks_in": size,
        "chains_out": chains,
        "max_gap_frames": config.link_max_gap_frames,
        "max_distance_m": max_distance,
        "reid": use_reid,
        "reid_min_cosine": config.reid_min_cosine if use_reid else None,
    }


def resolve_jerseys(
    summaries: dict[int, TrackSummary],
    config: RefinementConfig,
    groups: dict[int, int] | None = None,
    identity_reads: dict[int, JerseyTally] | None = None,
) -> dict[str, Any]:
    """Collapse jersey reads into one number per identity group.

    Each tracklet contributes one vote of unit mass, scaled by the square of
    its median box height: a number read off a 180 px crop is far more
    trustworthy than the same number read off a 70 px crop, and fragments of
    one player routinely disagree exactly along that axis. Detection count is
    deliberately not part of the weight -- a long tracklet of tiny crops is not
    better evidence than a short tracklet of large ones.

    ``identity_reads`` carries a second, independent opinion: a reader asked
    about the whole identity at once, keyed by group id (see
    ``scripts/reread_jerseys.py``). It answers several times per identity, so
    it arrives as a :class:`JerseyTally`, and it enters the same pool with
    mass ``identity_read_weight`` scaled by how much it agreed with itself.
    """
    groups = groups or {track_id: track_id for track_id in summaries}
    pooled: dict[int, Counter] = defaultdict(Counter)
    for track_id, summary in summaries.items():
        total_votes = sum(summary.jersey_votes.values())
        if not total_votes:
            continue
        weight = (max(summary.median_height, 1.0) / 100.0) ** 2
        for number, count in summary.jersey_votes.items():
            pooled[groups.get(track_id, track_id)][number] += weight * count / total_votes

    reread = 0
    ignored = 0
    for group, tally in (identity_reads or {}).items():
        if not tally.votes or tally.questions <= 0:
            continue
        if tally.consensus < config.identity_read_min_consensus:
            # The reader looked at the same player several times and named him
            # differently each time. That is not a weak opinion to be blended
            # in, it is no opinion at all.
            ignored += 1
            continue
        reread += 1
        # Divide by questions asked, not by answers given. An identity the
        # reader named once in six tries then contributes a sixth of the mass
        # of one it named six times out of six, and a scattered read falls back
        # to whatever the per-tracklet evidence already said.
        for number, count in tally.votes.items():
            pooled[group][str(number)] += config.identity_read_weight * count / tally.questions

    conflicts = 0
    if config.enforce_number_exclusivity:
        conflicts = _resolve_number_conflicts(pooled, summaries, groups)

    decided: dict[int, str] = {}
    for group, votes in pooled.items():
        if not votes:
            continue
        number, weighted = votes.most_common(1)[0]
        total = sum(votes.values())
        raw = sum(
            summary.jersey_votes[number]
            for track_id, summary in summaries.items()
            if groups.get(track_id, track_id) == group
        )
        tally = (identity_reads or {}).get(group)
        raw += tally.votes.get(number, 0) if tally else 0
        if raw >= config.jersey_min_votes and weighted / total >= config.jersey_min_share:
            decided[group] = number
    for track_id, summary in summaries.items():
        summary.jersey = decided.get(groups.get(track_id, track_id))
    return {
        "groups_with_reads": len(pooled),
        "groups_resolved": len(decided),
        "groups_with_identity_reread": reread,
        "identity_rereads_ignored": ignored,
        "identity_read_weight": config.identity_read_weight if reread else None,
        "number_conflicts_resolved": conflicts,
        "tracks_resolved": sum(1 for summary in summaries.values() if summary.jersey),
    }


def _identity_facts(
    summaries: dict[int, TrackSummary], groups: dict[int, int]
) -> dict[int, tuple[str | None, int, int, bool]]:
    """Per identity: team, frame span, and whether any of it is an outfield player."""
    facts: dict[int, tuple[str | None, int, int, bool]] = {}
    for track_id, summary in summaries.items():
        group = groups.get(track_id, track_id)
        team, start, end, outfield = facts.get(
            group, (summary.team, summary.start_frame, summary.end_frame, False)
        )
        facts[group] = (
            team or summary.team,
            min(start, summary.start_frame),
            max(end, summary.end_frame),
            outfield or summary.role == "player",
        )
    return facts


def _resolve_number_conflicts(
    pooled: dict[int, Counter], summaries: dict[int, TrackSummary], groups: dict[int, int]
) -> int:
    """Stop one number being worn by two players of the same team at once.

    Two identities that are on the pitch at the same moment, for the same team,
    cannot share a shirt number. When they claim one anyway the weaker claim is
    struck out and that identity falls back to its next-best number -- which is
    how five different players all reading as "10" collapses back to one.

    Identities that never overlap in time are left alone: they may well be two
    fragments of the same player that linking failed to join.
    """
    facts = _identity_facts(summaries, groups)
    resolved = 0
    while True:
        claims: dict[tuple[str | None, str], list[int]] = defaultdict(list)
        for group, votes in pooled.items():
            team, _, _, outfield = facts.get(group, (None, 0, 0, False))
            if not votes or not outfield or team is None:
                continue
            claims[(team, votes.most_common(1)[0][0])].append(group)

        struck = False
        for (_, number), contenders in claims.items():
            if len(contenders) < 2:
                continue
            contenders.sort(key=lambda group: -pooled[group][number])
            winner = contenders[0]
            _, winner_start, winner_end, _ = facts[winner]
            for loser in contenders[1:]:
                _, start, end, _ = facts[loser]
                if start > winner_end or end < winner_start:
                    continue  # never on the pitch together: possibly one player
                del pooled[loser][number]
                resolved += 1
                struck = True
        if not struck:
            return resolved


def refine_predictions(
    predictions: Sequence[Detection],
    frames: FrameLoader,
    config: RefinementConfig | None = None,
    embeddings: dict[int, np.ndarray] | None = None,
    identity_reads: dict[int, JerseyTally] | None = None,
) -> tuple[list[Detection], dict[str, Any]]:
    """Return attribute-refined copies of ``predictions`` plus an audit report."""
    config = config or RefinementConfig()
    summaries = summarise_tracks(predictions, frames, config)
    team_report = assign_teams(summaries, predictions, config)
    if config.link_tracklets:
        remap, link_report = link_tracklets(summaries, config, embeddings)
    else:
        remap, link_report = {}, {"links": 0}
    jersey_report = resolve_jerseys(summaries, config, remap, identity_reads)

    changed_role = 0
    changed_team = 0
    changed_jersey = 0
    refined: list[Detection] = []
    for detection in predictions:
        item = dict(detection)
        attributes = dict(item.get("attributes") or {})
        summary = summaries.get(item.get("track_id"))
        if summary is not None and attributes.get("role") in TRACKED_ROLES:
            if attributes.get("role") != summary.role:
                attributes["role"] = summary.role
                changed_role += 1
        if summary is not None and summary.role in PLAYER_ROLES:
            if summary.team is not None and attributes.get("team") != summary.team:
                attributes["team"] = summary.team
                changed_team += 1
            if attributes.get("role") == "player":
                previous = attributes.get("jersey")
                # Writing a number where the reader abstained is a two-sided
                # bet: the annotation is equally often "no readable number".
                # Propagating along a link is the safer half of that bet -- the
                # evidence comes from another view of the same player -- so it
                # is controlled separately from filling a tracklet's own gaps.
                if previous not in (None, ""):
                    emit = True
                elif summary.jersey_votes:
                    emit = config.jersey_fill
                else:
                    emit = config.jersey_propagate
                if summary.jersey is not None and emit:
                    if previous != summary.jersey:
                        changed_jersey += 1
                    attributes["jersey"] = summary.jersey
                elif summary.jersey is None and previous not in (None, ""):
                    attributes["jersey"] = None
                    changed_jersey += 1
            item["track_id"] = remap.get(item["track_id"], item["track_id"])
        item["attributes"] = attributes
        refined.append(item)

    report = {
        "detections": len(refined),
        "tracks": len(summaries),
        "team": team_report,
        "jersey": jersey_report,
        "linking": link_report,
        "role_labels_changed": changed_role,
        "team_labels_changed": changed_team,
        "jersey_labels_changed": changed_jersey,
        "config": {
            "frame_stride": config.frame_stride,
            "offside_rank": config.offside_rank,
            "reconcile_roles": config.reconcile_roles,
            "jersey_min_votes": config.jersey_min_votes,
            "jersey_min_share": config.jersey_min_share,
            "identity_read_weight": config.identity_read_weight,
            "identity_read_min_consensus": config.identity_read_min_consensus,
            "enforce_number_exclusivity": config.enforce_number_exclusivity,
            "jersey_fill": config.jersey_fill,
            "jersey_propagate": config.jersey_propagate,
            "link_tracklets": config.link_tracklets,
            "link_max_gap_frames": config.link_max_gap_frames,
            "link_max_distance_m": config.link_max_distance_m,
        },
    }
    return refined, report


class _TrackletStub:
    """Stand-in for the perception backend's ``Tracklet`` class.

    The backend dumps its per-detection re-identification features as a pickle
    of its own ``Tracklet`` objects. Rebuilding them here as plain attribute
    bags lets this process read those features without importing the backend.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class _TrackletUnpickler:
    @staticmethod
    def load(path: str) -> Any:
        import pickle

        class Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if name == "Tracklet":
                    return _TrackletStub
                return super().find_class(module, name)

        with open(path, "rb") as handle:
            return Unpickler(handle).load()


def detection_key(frame: int, x: float, y: float) -> tuple[int, float, float]:
    """The join key shared by the backend's tracklet dump and its predictions."""
    return int(frame), round(float(x), 1), round(float(y), 1)


def load_reid_detections(pickle_path: str) -> dict[tuple[int, float, float], np.ndarray]:
    """Per-detection re-identification vectors from the backend's tracklet dump.

    The dump predates the backend's own tracklet merging, so its ids do not
    line up with the ids in the exported predictions. Detections are joined on
    ``(frame, box x, box y)``, which the two files share exactly.
    """
    dump = _TrackletUnpickler.load(pickle_path)
    tracklets = dump.values() if isinstance(dump, dict) else dump
    features: dict[tuple[int, float, float], np.ndarray] = {}
    for tracklet in tracklets:
        boxes = getattr(tracklet, "bboxes", None)
        vectors = getattr(tracklet, "features", None)
        times = getattr(tracklet, "times", None)
        if not boxes or vectors is None or times is None:
            continue
        for frame, box, vector in zip(times, boxes, vectors, strict=False):
            features[detection_key(frame, box[0], box[1])] = np.asarray(vector, dtype=float)
    return features


def load_reid_embeddings(
    pickle_path: str, predictions: Sequence[Detection]
) -> dict[int, np.ndarray]:
    """Mean unit-norm re-identification embedding per predicted track."""
    features = load_reid_detections(pickle_path)
    pooled: dict[int, list[np.ndarray]] = defaultdict(list)
    for detection in predictions:
        if (detection.get("attributes") or {}).get("role") not in PLAYER_ROLES:
            continue
        box = detection["bbox_image"]
        vector = features.get(detection_key(detection["frame"], box["x"], box["y"]))
        if vector is not None:
            pooled[detection["track_id"]].append(vector)

    embeddings: dict[int, np.ndarray] = {}
    for track_id, vectors in pooled.items():
        mean = np.mean(np.asarray(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            embeddings[track_id] = mean / norm
    return embeddings


class ImageSequenceFrames:
    """Frame loader over a SoccerNet ``img1`` directory."""

    def __init__(self, directory: str, pattern: str = "{:06d}.jpg"):
        self.directory = directory
        self.pattern = pattern

    def __call__(self, frame_number: int) -> np.ndarray | None:
        import cv2

        return cv2.imread(f"{self.directory}/{self.pattern.format(frame_number)}")


class VideoFrames:
    """Frame loader over a video file, decoding sequentially where possible."""

    def __init__(self, path: str):
        import cv2

        self._capture = cv2.VideoCapture(path)
        self._position = 0

    def __call__(self, frame_number: int) -> np.ndarray | None:
        import cv2

        if frame_number - 1 != self._position:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            self._position = frame_number - 1
        ok, frame = self._capture.read()
        self._position += 1
        return frame if ok else None


def sequence_frame_loader(root: str, sequence: str) -> FrameLoader:
    """Prefer extracted frames, fall back to the encoded clip."""
    import os

    directory = os.path.join(root, sequence, "img1")
    if os.path.isdir(directory):
        return ImageSequenceFrames(directory)
    return VideoFrames(os.path.join(root, sequence, f"{sequence}.mp4"))


__all__ = [
    "ImageSequenceFrames",
    "RefinementConfig",
    "JerseyTally",
    "TrackSummary",
    "VideoFrames",
    "assign_teams",
    "demote_stranded_goalkeepers",
    "demote_surplus_referees",
    "refine_predictions",
    "detection_key",
    "link_tracklets",
    "load_reid_detections",
    "load_reid_embeddings",
    "resolve_jerseys",
    "sequence_frame_loader",
    "summarise_tracks",
    "torso_descriptor",
]
