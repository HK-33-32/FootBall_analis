"""Per-player match statistics, with the timecodes each number came from.

Every figure here is derived from the refined game state and nothing else, so
each one inherits the limits of what produced it. Three of those matter enough
to carry in the output rather than a footnote:

* a player is only measured while the camera can see him and the pitch can be
  calibrated, so "distance covered" means distance covered *on camera*, and the
  ``coverage`` field says how much of the clip that was;
* positions come from inverting a homography, so raw frame-to-frame jitter
  would be counted as running. Trajectories are speed-gated and smoothed by
  :mod:`football_intelligence.trajectories` before any distance is summed;
* possession is proximity to the ball, not a detected touch. It is reported as
  ``time_nearest_ball_s``, which is what it measures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .gamestate import PLAYER_ROLES, Detection
from .trajectories import (
    TrajectoryConfig,
    clean_ball_track,
    clean_player_track,
    path_length_m_within,
    speeds_m_s,
)


@dataclass(frozen=True)
class StatsConfig:
    """Thresholds behind the derived numbers."""

    fps: float = 25.0
    sprint_speed_m_s: float = 7.0
    sprint_min_duration_s: float = 0.6
    running_speed_m_s: float = 4.0
    possession_radius_m: float = 2.5
    segment_gap_frames: int = 12
    heatmap_columns: int = 14
    heatmap_rows: int = 9
    pitch_half_length_m: float = 52.5
    pitch_half_width_m: float = 34.0
    # Zone edges in m/s, the boundaries sports science conventionally uses.
    speed_zones: tuple[tuple[str, float, float], ...] = (
        ("walk", 0.0, 2.0),
        ("jog", 2.0, 4.0),
        ("run", 4.0, 5.5),
        ("high_intensity", 5.5, 7.0),
        ("sprint", 7.0, 99.0),
    )
    acceleration_m_s2: float = 2.0
    acceleration_min_duration_s: float = 0.24
    min_team_shape_players: int = 4


def _segments(frames: Sequence[int], gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of frames, allowing ``gap`` missing frames inside one."""
    if not frames:
        return []
    ordered = sorted(frames)
    spans = [[ordered[0], ordered[0]]]
    for frame in ordered[1:]:
        if frame - spans[-1][1] <= gap:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return [(low, high) for low, high in spans]


def _timecode(frame: int, fps: float) -> str:
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _span_payload(spans: Sequence[tuple[int, int]], fps: float) -> list[dict[str, Any]]:
    return [
        {
            "start_frame": low,
            "end_frame": high,
            "start_s": round(low / fps, 3),
            "end_s": round(high / fps, 3),
            "start_timecode": _timecode(low, fps),
            "end_timecode": _timecode(high, fps),
            "duration_s": round((high - low + 1) / fps, 3),
        }
        for low, high in spans
    ]


def _thirds(points: Sequence[tuple[float, float]], attacking: str, half_length: float) -> dict:
    """Share of samples in each third, named from the player's own direction."""
    if not points:
        return {"defensive": 0.0, "middle": 0.0, "attacking": 0.0}
    xs = np.asarray([point[0] for point in points], dtype=float)
    if attacking == "left":
        xs = -xs
    third = half_length / 3.0
    counts = {
        "defensive": float((xs < -third).mean()),
        "middle": float(((xs >= -third) & (xs <= third)).mean()),
        "attacking": float((xs > third).mean()),
    }
    return {key: round(value, 4) for key, value in counts.items()}


def _heatmap(points: Sequence[tuple[float, float]], config: StatsConfig) -> list[list[int]]:
    grid = np.zeros((config.heatmap_rows, config.heatmap_columns), dtype=int)
    for x, y in points:
        column = int(
            (x + config.pitch_half_length_m)
            / (2 * config.pitch_half_length_m)
            * config.heatmap_columns
        )
        row = int(
            (y + config.pitch_half_width_m) / (2 * config.pitch_half_width_m) * config.heatmap_rows
        )
        column = min(max(column, 0), config.heatmap_columns - 1)
        row = min(max(row, 0), config.heatmap_rows - 1)
        grid[row, column] += 1
    return grid.tolist()


def _sprints(
    frames: Sequence[int], speeds: Sequence[float], config: StatsConfig
) -> list[dict[str, Any]]:
    """Runs above the sprint threshold that last long enough to be a sprint."""
    spans: list[dict[str, Any]] = []
    start: int | None = None
    peak = 0.0
    for index, speed in enumerate(speeds):
        if speed >= config.sprint_speed_m_s:
            start = frames[index] if start is None else start
            peak = max(peak, speed)
            continue
        if start is not None:
            end = frames[index]
            if (end - start) / config.fps >= config.sprint_min_duration_s:
                spans.append({"start": start, "end": end, "peak": peak})
            start, peak = None, 0.0
    if start is not None and (frames[-1] - start) / config.fps >= config.sprint_min_duration_s:
        spans.append({"start": start, "end": frames[-1], "peak": peak})
    return [
        {
            **_span_payload([(span["start"], span["end"])], config.fps)[0],
            "top_speed_kmh": round(span["peak"] * 3.6, 1),
        }
        for span in spans
    ]


def _speed_zones(
    frames: Sequence[int],
    points: Sequence[tuple[float, float]],
    speeds: Sequence[float],
    config: StatsConfig,
    max_gap: int,
) -> dict[str, dict[str, float]]:
    """Time and distance spent in each speed band.

    A distance total on its own says how far a player travelled; the split says
    whether he walked it or ran it, which is what the number is usually wanted
    for.
    """
    result = {name: {"time_s": 0.0, "distance_m": 0.0} for name, _, _ in config.speed_zones}
    for index in range(1, len(frames)):
        step_frames = frames[index] - frames[index - 1]
        if step_frames > max_gap:
            continue
        step = float(np.hypot(*(np.subtract(points[index], points[index - 1]))))
        speed = speeds[index] if index < len(speeds) else 0.0
        for name, low, high in config.speed_zones:
            if low <= speed < high:
                result[name]["time_s"] += step_frames / config.fps
                result[name]["distance_m"] += step
                break
    return {
        name: {"time_s": round(values["time_s"], 2), "distance_m": round(values["distance_m"], 1)}
        for name, values in result.items()
    }


def _accelerations(
    frames: Sequence[int], speeds: Sequence[float], config: StatsConfig
) -> dict[str, Any]:
    """Sustained changes of pace, counted separately in each direction.

    Acceleration is a second derivative of a noisy position, so it is taken
    from the already-smoothed speed series over a baseline of several frames
    and only counted when it holds. It remains the shakiest figure here.
    """
    if len(frames) < 3:
        return {"accelerations": 0, "decelerations": 0, "peak_acceleration_m_s2": 0.0}
    span = max(1, int(round(config.acceleration_min_duration_s * config.fps)))
    events = {"accelerations": 0, "decelerations": 0}
    peak = 0.0
    index = 0
    while index + span < len(frames):
        gap = (frames[index + span] - frames[index]) / config.fps
        if gap <= 0 or frames[index + span] - frames[index] > span * 3:
            index += 1
            continue
        change = (speeds[index + span] - speeds[index]) / gap
        if abs(change) >= config.acceleration_m_s2:
            events["accelerations" if change > 0 else "decelerations"] += 1
            peak = max(peak, abs(change))
            index += span  # one burst is one event, not one per frame
            continue
        index += 1
    return {**events, "peak_acceleration_m_s2": round(peak, 2)}


def _team_shape(
    outfield_at: dict[int, list[tuple[int, tuple[float, float]]]],
    sides: dict[int, str | None],
    config: StatsConfig,
) -> dict[str, dict[str, float]]:
    """Width, depth, compactness and line height, per team, over the clip."""
    gathered: dict[str, dict[str, list[float]]] = {
        side: {"width": [], "depth": [], "compactness": [], "line": []}
        for side in ("left", "right")
    }
    for _, rows in outfield_at.items():
        by_side: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
        for track, point in rows:
            side = sides.get(track)
            if side in by_side:
                by_side[side].append(point)
        for side, points in by_side.items():
            if len(points) < config.min_team_shape_players:
                continue
            array = np.asarray(points, dtype=float)
            centre = array.mean(axis=0)
            gathered[side]["width"].append(float(array[:, 1].max() - array[:, 1].min()))
            gathered[side]["depth"].append(float(array[:, 0].max() - array[:, 0].min()))
            gathered[side]["compactness"].append(
                float(np.mean(np.linalg.norm(array - centre, axis=1)))
            )
            # Line height counted towards the goal this team attacks: the side
            # defending the left goal attacks towards +x, so positive means
            # advanced and negative means camped in its own half.
            gathered[side]["line"].append(float(centre[0]) * (1 if side == "left" else -1))
    shape: dict[str, dict[str, float]] = {}
    for side, values in gathered.items():
        if not values["width"]:
            shape[side] = {}
            continue
        shape[side] = {
            "frames_measured": len(values["width"]),
            "width_m": round(float(np.median(values["width"])), 1),
            "depth_m": round(float(np.median(values["depth"])), 1),
            "compactness_m": round(float(np.median(values["compactness"])), 1),
            "line_height_m": round(float(np.median(values["line"])), 1),
        }
    return shape


def _turnovers(
    nearest: dict[int, list[int]], sides: dict[int, str | None], config: StatsConfig
) -> dict[str, Any]:
    """How often the ball changed teams, and how long each spell lasted."""
    owner_at: dict[int, str] = {}
    for track, frames in nearest.items():
        side = sides.get(track)
        if side not in ("left", "right"):
            continue
        for frame in frames:
            owner_at[frame] = side
    spells: list[tuple[str, int, int]] = []
    for frame in sorted(owner_at):
        side = owner_at[frame]
        if spells and spells[-1][0] == side and frame - spells[-1][2] <= config.segment_gap_frames:
            spells[-1] = (side, spells[-1][1], frame)
        else:
            spells.append((side, frame, frame))
    durations = [(end - start + 1) / config.fps for _, start, end in spells]
    return {
        "spells": len(spells),
        "turnovers": max(0, len(spells) - 1),
        "median_spell_s": round(float(np.median(durations)), 2) if durations else 0.0,
        "longest_spell_s": round(max(durations), 2) if durations else 0.0,
    }


def match_statistics(
    predictions: Sequence[Detection],
    config: StatsConfig | None = None,
    trajectory_config: TrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Per-identity statistics plus the ball track and team totals."""
    config = config or StatsConfig()
    trajectory_config = trajectory_config or TrajectoryConfig(fps=config.fps)

    people: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    roles: dict[int, Counter] = defaultdict(Counter)
    teams: dict[int, Counter] = defaultdict(Counter)
    jerseys: dict[int, Counter] = defaultdict(Counter)
    ball: list[tuple[int, tuple[float, float]]] = []
    all_frames: set[int] = set()

    for detection in predictions:
        attributes = detection.get("attributes") or {}
        pitch = detection.get("bbox_pitch") or {}
        if "x_bottom_middle" not in pitch:
            continue
        frame = int(detection["frame"])
        all_frames.add(frame)
        point = (float(pitch["x_bottom_middle"]), float(pitch["y_bottom_middle"]))
        role = attributes.get("role")
        if role == "ball":
            ball.append((frame, point))
            continue
        if role not in PLAYER_ROLES and role != "referee":
            continue
        track = int(detection["track_id"])
        people[track].append((frame, point))
        roles[track][role] += 1
        if attributes.get("team"):
            teams[track][attributes["team"]] += 1
        if attributes.get("jersey") not in (None, ""):
            jerseys[track][str(attributes["jersey"])] += 1

    ball_track = clean_ball_track(ball, trajectory_config)
    ball_at = dict(zip(ball_track["frames"], ball_track["points"], strict=True))

    cleaned: dict[int, tuple[list[int], list[tuple[float, float]]]] = {}
    for track, observations in people.items():
        cleaned[track] = clean_player_track(observations, trajectory_config)

    # possession: at each frame, the single closest person within the radius.
    # Indexed by frame rather than scanned per track: over a full match the
    # naive form is frames x tracks, which is hundreds of millions of lookups
    # for a number that only needs the handful of people actually on screen.
    outfield_at: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for track, (track_frames, track_points) in cleaned.items():
        if roles[track].most_common(1)[0][0] == "referee":
            continue
        for frame, point in zip(track_frames, track_points, strict=True):
            outfield_at[frame].append((track, point))

    nearest: dict[int, list[int]] = defaultdict(list)
    for frame, ball_point in ball_at.items():
        best_track, best_distance = None, config.possession_radius_m
        for track, point in outfield_at.get(frame, ()):
            distance = float(np.hypot(point[0] - ball_point[0], point[1] - ball_point[1]))
            if distance < best_distance:
                best_track, best_distance = track, distance
        if best_track is not None:
            nearest[best_track].append(frame)

    first_frame, last_frame = (min(all_frames), max(all_frames)) if all_frames else (0, 0)
    clip_frames = max(1, last_frame - first_frame + 1)

    sides = {
        track: (teams[track].most_common(1)[0][0] if teams[track] else None) for track in people
    }
    players: list[dict[str, Any]] = []
    for track, (frames, points) in sorted(cleaned.items(), key=lambda kv: -len(kv[1][0])):
        if not frames:
            continue
        role = roles[track].most_common(1)[0][0]
        team = sides[track]
        speeds = speeds_m_s(frames, points, trajectory_config)
        distance = path_length_m_within(
            frames, points, trajectory_config.max_interpolated_gap_frames
        )
        moving = [speed for speed in speeds if speed >= config.running_speed_m_s]
        possession_frames = nearest.get(track, [])
        players.append(
            {
                "identity": track,
                "role": role,
                "team": team,
                "jersey": jerseys[track].most_common(1)[0][0] if jerseys[track] else None,
                "detections": len(people[track]),
                "samples_used": len(frames),
                "coverage": round(len(frames) / clip_frames, 4),
                "visible_segments": _span_payload(
                    _segments(frames, config.segment_gap_frames), config.fps
                ),
                "time_on_camera_s": round(len(frames) / config.fps, 2),
                "distance_m": round(distance, 1),
                "distance_per_minute_m": round(distance / (len(frames) / config.fps / 60), 1)
                if frames
                else 0.0,
                "top_speed_kmh": round(max(speeds, default=0.0) * 3.6, 1),
                "average_speed_kmh": round(float(np.mean(speeds)) * 3.6, 1) if speeds else 0.0,
                "time_running_s": round(len(moving) / config.fps, 2),
                "sprints": _sprints(frames, speeds, config),
                "thirds": _thirds(points, team or "right", config.pitch_half_length_m),
                "mean_position": [
                    round(float(np.mean([point[0] for point in points])), 2),
                    round(float(np.mean([point[1] for point in points])), 2),
                ],
                "heatmap": _heatmap(points, config),
                "time_nearest_ball_s": round(len(possession_frames) / config.fps, 2),
                "ball_segments": _span_payload(
                    _segments(possession_frames, config.segment_gap_frames), config.fps
                ),
                "ball_episodes": len(_segments(possession_frames, config.segment_gap_frames)),
                "speed_zones": _speed_zones(
                    frames,
                    points,
                    speeds,
                    config,
                    trajectory_config.max_interpolated_gap_frames,
                ),
                **_accelerations(frames, speeds, config),
                "top_speed_at": _timecode(
                    frames[int(np.argmax(speeds))] if speeds else frames[0], config.fps
                ),
                "width_used_m": round(
                    float(np.percentile([point[1] for point in points], 90))
                    - float(np.percentile([point[1] for point in points], 10)),
                    1,
                ),
                "depth_used_m": round(
                    float(np.percentile([point[0] for point in points], 90))
                    - float(np.percentile([point[0] for point in points], 10)),
                    1,
                ),
            }
        )

    ball_speeds = speeds_m_s(ball_track["frames"], ball_track["points"], trajectory_config)
    ball_top_speed = max(ball_speeds, default=0.0)

    shape = _team_shape(outfield_at, sides, config)
    possession_flow = _turnovers(nearest, sides, config)

    team_totals: dict[str, Any] = {}
    for side in ("left", "right"):
        members = [player for player in players if player["team"] == side]
        possession = sum(player["time_nearest_ball_s"] for player in members)
        zones: dict[str, float] = {}
        for player in members:
            for name, values in player["speed_zones"].items():
                zones[name] = round(zones.get(name, 0.0) + values["distance_m"], 1)
        team_totals[side] = {
            "identities": len(members),
            "distance_m": round(sum(player["distance_m"] for player in members), 1),
            "time_nearest_ball_s": round(possession, 2),
            "sprints": sum(len(player["sprints"]) for player in members),
            "accelerations": sum(player["accelerations"] for player in members),
            "decelerations": sum(player["decelerations"] for player in members),
            "distance_by_zone_m": zones,
            "shape": shape.get(side, {}),
        }
    contested = sum(entry["time_nearest_ball_s"] for entry in team_totals.values())
    for side in team_totals:
        team_totals[side]["possession_share"] = (
            round(team_totals[side]["time_nearest_ball_s"] / contested, 4) if contested else None
        )

    return {
        "fps": config.fps,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frames_with_game_state": len(all_frames),
        "clip_frames": clip_frames,
        "coverage": round(len(all_frames) / clip_frames, 4),
        "ball": {
            "top_speed_kmh": round(ball_top_speed * 3.6, 1),
            "observed": ball_track["observed"],
            "kept": ball_track["kept"],
            "off_pitch_dropped": ball_track["off_pitch_dropped"],
            "unreachable_dropped": ball_track["unreachable_dropped"],
            "interpolated": ball_track["interpolated_count"],
            "distance_m": round(
                path_length_m_within(
                    ball_track["frames"],
                    ball_track["points"],
                    trajectory_config.max_interpolated_gap_frames,
                ),
                1,
            ),
        },
        "teams": team_totals,
        "possession_flow": possession_flow,
        "ball_tracked_share": round(len(ball_at) / max(len(all_frames), 1), 4),
        "players": players,
        "thresholds": {
            "sprint_speed_kmh": round(config.sprint_speed_m_s * 3.6, 1),
            "running_speed_kmh": round(config.running_speed_m_s * 3.6, 1),
            "possession_radius_m": config.possession_radius_m,
            "ball_max_speed_kmh": round(trajectory_config.ball_max_speed_m_s * 3.6, 1),
            "player_max_speed_kmh": round(trajectory_config.player_max_speed_m_s * 3.6, 1),
            "acceleration_m_s2": config.acceleration_m_s2,
            "speed_zones_kmh": {
                name: [round(low * 3.6, 1), round(high * 3.6, 1)]
                for name, low, high in config.speed_zones
            },
        },
    }


def timeline(
    predictions: Sequence[Detection],
    ball_frames: Sequence[int],
    ball_points: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Compact per-frame positions for playback, as parallel arrays."""
    by_frame: dict[int, list[list[Any]]] = defaultdict(list)
    for detection in predictions:
        attributes = detection.get("attributes") or {}
        pitch = detection.get("bbox_pitch") or {}
        role = attributes.get("role")
        if "x_bottom_middle" not in pitch or role == "ball" or role is None:
            continue
        by_frame[int(detection["frame"])].append(
            [
                int(detection["track_id"]),
                round(float(pitch["x_bottom_middle"]), 2),
                round(float(pitch["y_bottom_middle"]), 2),
            ]
        )
    ball_at = {
        frame: [round(point[0], 2), round(point[1], 2)]
        for frame, point in zip(ball_frames, ball_points, strict=True)
    }
    return {
        "frames": sorted(by_frame),
        "people": {str(frame): rows for frame, rows in sorted(by_frame.items())},
        "ball": {str(frame): point for frame, point in sorted(ball_at.items())},
    }


__all__ = ["StatsConfig", "match_statistics", "timeline"]
