"""Making pitch trajectories physical.

Detections are placed on the pitch by inverting a per-frame homography. Small
image errors become large pitch errors far from the camera, and a single false
positive -- a boot, a line marking, a distant white object -- lands anywhere at
all. Read literally, the resulting ball track teleports: measured on broadcast
footage, a third of its frame-to-frame steps imply speeds above 40 m/s and the
worst exceeds 2000 m/s, against roughly 35 m/s for a struck ball.

Two tools here, both working in metres on the pitch rather than in pixels:

* :func:`longest_reachable_run` keeps the largest subset of detections that a
  real ball could have visited in order, and drops the rest;
* :func:`smooth_track` removes the residual jitter that would otherwise be
  counted as distance covered.

Nothing here touches identity attributes, and the ball is excluded from
GS-HOTA scoring, so none of it moves the benchmark. It exists because a game
state that cannot be read as motion is not a game state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class TrajectoryConfig:
    """Physical limits a real trajectory has to respect."""

    pitch_half_length_m: float = 52.5
    pitch_half_width_m: float = 34.0
    out_of_pitch_margin_m: float = 3.0
    ball_max_speed_m_s: float = 30.0
    player_max_speed_m_s: float = 11.0
    step_tolerance_m: float = 0.25
    smoothing_window: int = 5
    mean_window: int = 11
    speed_window: int = 4
    max_interpolated_gap_frames: int = 12
    reach_window_frames: int = 50
    segment_gap_frames: int = 50
    fps: float = 25.0


def inside_pitch(point: Point, config: TrajectoryConfig) -> bool:
    """Is this position on the pitch, allowing for calibration slack?"""
    return (
        abs(point[0]) <= config.pitch_half_length_m + config.out_of_pitch_margin_m
        and abs(point[1]) <= config.pitch_half_width_m + config.out_of_pitch_margin_m
    )


def longest_reachable_run(
    frames: Sequence[int], points: Sequence[Point], max_speed_m_s: float, config: TrajectoryConfig
) -> list[int]:
    """Indices of the largest subsequence a body moving at that speed could visit.

    Dropping every step that looks too fast would also drop the good detection
    that follows a bad one, and chaining greedily lets one outlier drag the
    whole track after it. Choosing the *longest* self-consistent subsequence
    instead lets the outliers fall out on their own: a spurious detection has
    nothing before or after it that it can be reached from.
    """
    count = len(frames)
    if count == 0:
        return []
    best = [1] * count
    previous = [-1] * count
    frame_array = np.asarray(frames, dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    # Only look back as far as a body could plausibly have come from. Scanning
    # every earlier detection is quadratic: invisible on a 30-second clip, and
    # twelve hours on a 90-minute one.
    window = max(1, config.reach_window_frames)
    first = 0
    for later in range(count):
        while frames[later] - frames[first] > window:
            first += 1
        if first == later:
            continue
        gaps = (frame_array[later] - frame_array[first:later]) / config.fps
        delta = point_array[later] - point_array[first:later]
        distances = np.hypot(delta[:, 0], delta[:, 1])
        reachable = (gaps > 0) & (distances <= max_speed_m_s * gaps + config.step_tolerance_m)
        scores = np.where(reachable, best[first:later], 0)
        # argmax retains the original earliest-predecessor tie-breaking.
        winner = int(np.argmax(scores))
        if scores[winner] > 0:
            best[later] = int(scores[winner]) + 1
            previous[later] = first + winner
    end = int(np.argmax(best))
    chain = []
    while end != -1:
        chain.append(end)
        end = previous[end]
    return chain[::-1]


def _window_bounds(frames: Sequence[int], index: int, half: int) -> tuple[int, int]:
    """Neighbours within ``half`` *frames* of this sample, not ``half`` samples.

    A track with a hole in it has samples that are adjacent in the list and
    seconds apart on the clock. Averaging across that hole invents a position
    between two places the player never travelled between, and the invented
    step is then read as a sprint.
    """
    low = index
    while low > 0 and frames[index] - frames[low - 1] <= half:
        low -= 1
    high = index
    while high + 1 < len(frames) and frames[high + 1] - frames[index] <= half:
        high += 1
    return low, high + 1


def smooth_track(
    frames: Sequence[int], points: Sequence[Point], config: TrajectoryConfig
) -> list[Point]:
    """Median first, then mean, over frame-aligned windows.

    The median removes the spikes a mean would smear into its neighbours; the
    mean then removes the residual jitter, which matters because that jitter is
    otherwise summed as distance covered. Without the second pass a referee
    jogging for eight seconds "runs" 77 m.
    """
    if not points:
        return []
    array = np.asarray(points, dtype=float)
    for window, statistic in (
        (config.smoothing_window, np.median),
        (config.mean_window, np.mean),
    ):
        if window <= 1:
            continue
        half = (max(1, window | 1)) // 2
        updated = np.empty_like(array)
        for index in range(len(array)):
            low, high = _window_bounds(frames, index, half)
            updated[index] = statistic(array[low:high], axis=0)
        array = updated
    return [(float(x), float(y)) for x, y in array]


def interpolate_gaps(
    frames: Sequence[int], points: Sequence[Point], config: TrajectoryConfig
) -> tuple[list[int], list[Point], set[int]]:
    """Fill short gaps by straight-line motion, and say which frames were filled."""
    if not frames:
        return [], [], set()
    out_frames = [frames[0]]
    out_points = [points[0]]
    filled: set[int] = set()
    for index in range(1, len(frames)):
        gap = frames[index] - frames[index - 1]
        if 1 < gap <= config.max_interpolated_gap_frames:
            start, end = np.asarray(points[index - 1]), np.asarray(points[index])
            for step in range(1, gap):
                fraction = step / gap
                frame = frames[index - 1] + step
                out_frames.append(frame)
                out_points.append(tuple(start + (end - start) * fraction))
                filled.add(frame)
        out_frames.append(frames[index])
        out_points.append(points[index])
    return out_frames, [(float(x), float(y)) for x, y in out_points], filled


def split_segments(frames: Sequence[int], config: TrajectoryConfig) -> list[tuple[int, int]]:
    """Index ranges separated by a gap no trajectory should be chained across."""
    if not frames:
        return []
    bounds = []
    start = 0
    for index in range(1, len(frames)):
        if frames[index] - frames[index - 1] > config.segment_gap_frames:
            bounds.append((start, index))
            start = index
    bounds.append((start, len(frames)))
    return bounds


def clean_ball_track(
    observations: Sequence[tuple[int, Point]], config: TrajectoryConfig | None = None
) -> dict[str, object]:
    """Turn raw ball detections into a trajectory that obeys the laws of motion."""
    config = config or TrajectoryConfig()
    ordered = sorted(observations)
    kept_on_pitch = [(frame, point) for frame, point in ordered if inside_pitch(point, config)]
    frames = [frame for frame, _ in kept_on_pitch]
    points = [point for _, point in kept_on_pitch]

    # A ball track over a whole match is not one chain: the broadcast cuts away
    # and comes back. Each stretch is cleaned on its own and all of them are
    # kept -- taking the single longest chain would discard the rest of the game.
    out_frames: list[int] = []
    out_points: list[Point] = []
    filled: set[int] = set()
    kept_total = 0
    for begin, stop in split_segments(frames, config):
        part_frames = frames[begin:stop]
        part_points = points[begin:stop]
        run = longest_reachable_run(part_frames, part_points, config.ball_max_speed_m_s, config)
        part_frames = [part_frames[index] for index in run]
        part_points = [part_points[index] for index in run]
        # Interpolate before smoothing, so the median window spans neighbouring
        # moments rather than whichever samples happened to survive; then run the
        # reachability test once more, because smoothing moves points and the
        # guarantee has to hold for what is actually returned.
        part_frames, part_points, part_filled = interpolate_gaps(part_frames, part_points, config)
        part_points = smooth_track(part_frames, part_points, config)
        final = longest_reachable_run(part_frames, part_points, config.ball_max_speed_m_s, config)
        part_frames = [part_frames[index] for index in final]
        part_points = [part_points[index] for index in final]
        kept_total += len(final)
        filled |= {frame for frame in part_frames if frame in part_filled}
        out_frames.extend(part_frames)
        out_points.extend(part_points)

    return {
        "frames": out_frames,
        "points": out_points,
        "interpolated": sorted(filled),
        "observed": len(ordered),
        "off_pitch_dropped": len(ordered) - len(kept_on_pitch),
        "unreachable_dropped": len(kept_on_pitch) - kept_total,
        "kept": kept_total,
        "interpolated_count": len(filled),
    }


def clean_player_track(
    observations: Sequence[tuple[int, Point]], config: TrajectoryConfig | None = None
) -> tuple[list[int], list[Point]]:
    """The same treatment for a player, at a human top speed."""
    config = config or TrajectoryConfig()
    ordered = sorted(observations)
    frames = [frame for frame, _ in ordered]
    points = [point for _, point in ordered]
    out_frames: list[int] = []
    out_points: list[Point] = []
    for begin, stop in split_segments(frames, config):
        part_frames = frames[begin:stop]
        part_points = points[begin:stop]
        if len(part_frames) > 2:
            run = longest_reachable_run(
                part_frames, part_points, config.player_max_speed_m_s, config
            )
            part_frames = [part_frames[index] for index in run]
            part_points = [part_points[index] for index in run]
        out_frames.extend(part_frames)
        out_points.extend(smooth_track(part_frames, part_points, config))
    return out_frames, out_points


def path_length_m(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=float)
    return float(np.hypot(*(array[1:] - array[:-1]).T).sum())


def speeds_m_s(
    frames: Sequence[int], points: Sequence[Point], config: TrajectoryConfig
) -> list[float]:
    """Speed at each sample, measured across a frame-aligned baseline.

    Differencing consecutive frames divides a position error of a few
    centimetres by 40 ms and calls the result a sprint, so the baseline spans
    ``speed_window`` frames either side. Samples with no neighbour inside that
    baseline sit next to a hole in the track and get no speed at all rather
    than a fabricated one.
    """
    if len(frames) < 2:
        return []
    span = max(1, config.speed_window)
    result = []
    for index in range(len(frames)):
        low, high = _window_bounds(frames, index, span)
        high -= 1
        gap = (frames[high] - frames[low]) / config.fps
        if gap <= 0:
            result.append(0.0)
            continue
        distance = float(
            np.hypot(points[high][0] - points[low][0], points[high][1] - points[low][1])
        )
        result.append(distance / gap)
    return result


def path_length_m_within(
    frames: Sequence[int], points: Sequence[Point], max_gap_frames: int
) -> float:
    """Distance travelled, not counting jumps across holes in the track."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(points)):
        if frames[index] - frames[index - 1] > max_gap_frames:
            continue
        total += float(np.hypot(*(np.subtract(points[index], points[index - 1]))))
    return total


__all__ = [
    "TrajectoryConfig",
    "clean_ball_track",
    "clean_player_track",
    "inside_pitch",
    "interpolate_gaps",
    "longest_reachable_run",
    "path_length_m",
    "path_length_m_within",
    "smooth_track",
    "split_segments",
    "speeds_m_s",
]
