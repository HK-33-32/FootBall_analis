from __future__ import annotations

import numpy as np

from football_intelligence.longmatch import ChunkConfig, merge_chunks, plan_chunks, rebase
from football_intelligence.playability import PlayabilityConfig, grass_share, playable_segments


def _segment(start, end):
    return {"start_s": start, "end_s": end, "duration_s": end - start}


def test_a_long_stretch_is_cut_into_bounded_jobs():
    chunks = plan_chunks([_segment(0, 150)], ChunkConfig(max_chunk_s=60))
    assert len(chunks) == 3
    assert all(chunk["duration_s"] <= 60.001 for chunk in chunks)
    assert chunks[0]["start_s"] == 0
    assert abs(chunks[-1]["end_s"] - 150) < 0.01
    assert [chunk["name"] for chunk in chunks] == ["chunk0000", "chunk0001", "chunk0002"]


def test_chunks_carry_the_match_clock_in_their_frame_numbers():
    chunks = plan_chunks([_segment(0, 10), _segment(40, 50)], ChunkConfig(max_chunk_s=60, fps=25))
    assert chunks[0]["start_frame"] == 1
    assert chunks[1]["start_frame"] == 40 * 25 + 1


def test_stretches_too_short_to_analyse_are_dropped():
    assert plan_chunks([_segment(0, 1.0)], ChunkConfig(min_chunk_s=2.0)) == []


def _detection(frame, track):
    return {
        "frame": frame,
        "track_id": track,
        "bbox_image": {"x": 1.0, "y": 1.0, "w": 10.0, "h": 20.0},
        "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
        "attributes": {"role": "player", "team": "left", "jersey": None},
    }


def test_rebasing_moves_a_chunk_onto_the_match_clock():
    chunk = {"index": 2, "name": "chunk0002", "start_frame": 501}
    rows = rebase([_detection(1, 7), _detection(2, 7)], chunk, ChunkConfig())
    assert [row["frame"] for row in rows] == [501, 502]
    assert {row["track_id"] for row in rows} == {200_007}
    assert {row["chunk"] for row in rows} == {"chunk0002"}


def test_merging_keeps_identities_from_different_chunks_apart():
    """The same tracker id in two chunks is two different players."""
    first = {"index": 0, "name": "chunk0000", "start_frame": 1, "start_s": 0, "end_s": 4}
    second = {"index": 1, "name": "chunk0001", "start_frame": 101, "start_s": 4, "end_s": 8}
    merged = merge_chunks(
        [(first, [_detection(1, 3)]), (second, [_detection(1, 3)])], ChunkConfig()
    )
    assert merged["identities"] == 2
    assert [row["frame"] for row in merged["predictions"]] == [1, 101]
    assert len(merged["chunks"]) == 2


def test_merged_output_is_ordered_by_the_match_clock():
    late = {"index": 1, "name": "chunk0001", "start_frame": 201, "start_s": 8, "end_s": 12}
    early = {"index": 0, "name": "chunk0000", "start_frame": 1, "start_s": 0, "end_s": 4}
    merged = merge_chunks([(late, [_detection(1, 1)]), (early, [_detection(1, 1)])], ChunkConfig())
    frames = [row["frame"] for row in merged["predictions"]]
    assert frames == sorted(frames)


def _frame(colour):
    return np.full((180, 320, 3), colour, dtype=np.uint8)


def test_grass_is_recognised_and_a_close_up_is_not():
    config = PlayabilityConfig()
    pitch = _frame((60, 160, 60))  # BGR green
    assert grass_share(pitch, config) > 0.9
    skin = _frame((120, 150, 200))
    assert grass_share(skin, config) < 0.1


def test_segments_open_on_clear_grass_and_survive_one_dark_sample():
    config = PlayabilityConfig(stride_frames=5, min_segment_s=0.2, pad_s=0, merge_gap_s=0)
    shares = [0.0, 0.0, 0.7, 0.7, 0.42, 0.7, 0.7, 0.0, 0.0]
    scan = {
        "fps": 25.0,
        "frames": len(shares) * 5,
        "stride": 5,
        "samples": [(index * 5, share) for index, share in enumerate(shares)],
    }
    segments = playable_segments(scan, config)
    assert len(segments) == 1  # the 0.42 sample does not close the stretch
    assert segments[0]["start_frame"] == 10
    assert segments[0]["end_frame"] == 30


def test_nothing_playable_yields_nothing():
    config = PlayabilityConfig()
    scan = {"fps": 25.0, "frames": 100, "stride": 5, "samples": [(i * 5, 0.05) for i in range(20)]}
    assert playable_segments(scan, config) == []
