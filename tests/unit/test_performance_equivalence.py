"""Differential tests against the pre-optimization algorithms, not accuracy tuning."""

import cv2
import numpy as np
import pytest
from scipy.stats import spearmanr

from football_intelligence.calibration import _frame_geometry
from football_intelligence.gamestate import VideoFrames
from football_intelligence.trajectories import TrajectoryConfig, longest_reachable_run


def legacy_longest_run(frames, points, speed, config):
    if not frames:
        return []
    best, previous = [1] * len(frames), [-1] * len(frames)
    for later in range(len(frames)):
        for earlier in range(later):
            gap_frames = frames[later] - frames[earlier]
            if not 0 < gap_frames <= max(1, config.reach_window_frames):
                continue
            delta = np.subtract(points[later], points[earlier])
            distance = np.hypot(*delta)
            if distance <= speed * (gap_frames / config.fps) + config.step_tolerance_m:
                if best[earlier] + 1 > best[later]:
                    best[later], previous[later] = best[earlier] + 1, earlier
    end, chain = int(np.argmax(best)), []
    while end != -1:
        chain.append(end)
        end = previous[end]
    return chain[::-1]


@pytest.mark.parametrize("seed", range(10))
def test_vectorized_path_preserves_duplicates_gaps_and_ties(seed):
    rng = np.random.default_rng(seed)
    frames = np.cumsum(rng.choice([0, 1, 2, 50, 51], 120)).tolist()
    points = rng.integers(-20, 20, (120, 2)).astype(float).tolist()
    config = TrajectoryConfig()
    assert longest_reachable_run(frames, points, 30, config) == legacy_longest_run(
        frames, points, 30, config
    )


def test_rank_correlation_is_identical_with_ties_and_constant_axes():
    rng = np.random.default_rng(7)
    for _ in range(30):
        rows = rng.integers(0, 10, (12, 3)).astype(float)
        a = float(spearmanr(rows[:, 0], rows[:, 1]).statistic)
        b = float(spearmanr(rows[:, 0], rows[:, 2]).statistic)
        assert _frame_geometry(rows)[0] == (a if abs(a) >= abs(b) else b)
    assert _frame_geometry([(1, 0, 0), (1, 2, 3)])[0] == 0


def test_video_loader_preserves_pixels_for_sequential_backward_and_sparse_reads(tmp_path):
    video = tmp_path / "test_only_frames.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    assert writer.isOpened()
    rng = np.random.default_rng(11)
    for _ in range(60):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    current, reference = VideoFrames(str(video)), VideoFrames(str(video), sequential_gap_frames=0)
    try:
        for index in [1, 3, 5, 7, 7, 4, 6, 59, 60, 1]:
            assert np.array_equal(current(index), reference(index))
        assert current(0) is None
        assert current(61) is None
        assert np.array_equal(current(2), reference(2))
    finally:
        current.close()
        reference.close()
