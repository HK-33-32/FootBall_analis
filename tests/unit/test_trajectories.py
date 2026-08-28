from __future__ import annotations

import numpy as np

from football_intelligence.trajectories import (
    TrajectoryConfig,
    clean_ball_track,
    inside_pitch,
    interpolate_gaps,
    longest_reachable_run,
    path_length_m_within,
    smooth_track,
    speeds_m_s,
)


def _straight(count: int, speed: float, fps: float = 25.0):
    frames = list(range(1, count + 1))
    points = [(index * speed / fps, 0.0) for index in range(count)]
    return frames, points


def test_a_teleport_is_dropped_and_the_real_path_survives():
    frames, points = _straight(20, 8.0)
    points[10] = (400.0, 400.0)  # one detection on another continent
    kept = longest_reachable_run(frames, points, 12.0, TrajectoryConfig())
    assert len(kept) == 19
    assert 10 not in kept


def test_the_longest_run_wins_over_a_run_that_starts_earlier():
    """Two mutually inconsistent stories; the one with more support is kept."""
    frames = [1, 2, 3, 20, 21, 22, 23, 24]
    points = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0)] + [(80.0 + i * 0.2, 0.0) for i in range(5)]
    kept = longest_reachable_run(frames, points, 8.0, TrajectoryConfig())
    assert [frames[index] for index in kept] == [20, 21, 22, 23, 24]


def test_off_pitch_detections_are_rejected():
    config = TrajectoryConfig()
    assert inside_pitch((52.0, 30.0), config)
    assert inside_pitch((54.0, 35.0), config)  # inside the calibration margin
    assert not inside_pitch((62.0, 0.0), config)
    assert not inside_pitch((0.0, 38.0), config)


def test_smoothing_does_not_average_across_a_hole_in_the_track():
    """The samples are neighbours in the list and seconds apart on the clock."""
    frames = [1, 2, 3, 500, 501, 502]
    points = [(0.0, 0.0)] * 3 + [(40.0, 0.0)] * 3
    smoothed = smooth_track(frames, points, TrajectoryConfig())
    assert all(abs(x) < 1e-6 for x, _ in smoothed[:3])
    assert all(abs(x - 40.0) < 1e-6 for x, _ in smoothed[3:])


def test_speed_is_zero_across_a_hole_rather_than_fabricated():
    frames = [1, 2, 3, 500, 501, 502]
    points = [(0.0, 0.0)] * 3 + [(40.0, 0.0)] * 3
    assert max(speeds_m_s(frames, points, TrajectoryConfig())) < 1e-6


def test_distance_ignores_the_jump_across_a_hole():
    frames = [1, 2, 500, 501]
    points = [(0.0, 0.0), (1.0, 0.0), (40.0, 0.0), (41.0, 0.0)]
    assert path_length_m_within(frames, points, 12) == 2.0


def test_speed_over_a_baseline_survives_position_jitter():
    """Frame-to-frame differencing turns centimetres of noise into a sprint."""
    rng = np.random.default_rng(3)
    frames = list(range(1, 61))
    points = [
        (index * 2.0 / 25.0 + rng.normal(0, 0.12), rng.normal(0, 0.12)) for index in range(60)
    ]
    naive = TrajectoryConfig(speed_window=1, smoothing_window=1, mean_window=1)
    windowed = TrajectoryConfig(speed_window=4, smoothing_window=1, mean_window=1)
    raw = speeds_m_s(frames, points, naive)
    smoothed = speeds_m_s(frames, points, windowed)
    assert max(raw) > 6.0  # the walker is doing 2 m/s
    assert max(smoothed) < 0.6 * max(raw)
    assert 1.5 < sorted(smoothed)[len(smoothed) // 2] < 2.5


def test_short_gaps_are_interpolated_and_long_ones_are_not():
    config = TrajectoryConfig(max_interpolated_gap_frames=5)
    frames, points, filled = interpolate_gaps(
        [1, 4, 40], [(0.0, 0.0), (3.0, 0.0), (9.0, 0.0)], config
    )
    assert frames == [1, 2, 3, 4, 40]
    assert filled == {2, 3}
    assert points[1] == (1.0, 0.0)


def test_clean_ball_track_reports_what_it_threw_away():
    observations = [(frame, (frame * 0.4, 0.0)) for frame in range(1, 41)]
    observations.append((15, (300.0, 300.0)))  # far off the pitch
    observations.append((22, (10.0, -30.0)))  # on the pitch but unreachable
    result = clean_ball_track(observations, TrajectoryConfig())
    assert result["off_pitch_dropped"] == 1
    assert result["unreachable_dropped"] >= 1
    speeds = speeds_m_s(result["frames"], result["points"], TrajectoryConfig())
    assert max(speeds) < 40.0  # nothing left that a ball could not have done
