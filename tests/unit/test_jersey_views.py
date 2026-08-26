from __future__ import annotations

import numpy as np

from football_intelligence.jersey_views import (
    JerseyViewConfig,
    collect_identity_views,
    views_manifest,
)

KIT = (40, 40, 150)


def _frame_with(boxes: dict[tuple[int, int], tuple[int, int, int]]) -> np.ndarray:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 60, size=(400, 600, 3), dtype=np.uint8)
    for (x, y), colour in boxes.items():
        frame[y : y + 120, x : x + 50] = colour
    return frame


def _detection(frame: int, track_id: int, x: int, height: float = 120.0) -> dict:
    return {
        "frame": frame,
        "track_id": track_id,
        "bbox_image": {"x": float(x), "y": 20.0, "w": 50.0, "h": height},
        "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
        "attributes": {"role": "player", "team": "left", "jersey": None},
    }


def _clip(track_of_frame) -> tuple[list[dict], dict[int, np.ndarray]]:
    predictions: list[dict] = []
    frames: dict[int, np.ndarray] = {}
    for frame_number in range(1, 201):
        track_id = track_of_frame(frame_number)
        # boxes grow towards the middle of the clip, as a player nearing camera
        height = 120.0 - abs(100 - frame_number) * 0.4
        predictions.append(_detection(frame_number, track_id, 100, height))
        frames[frame_number] = _frame_with({(100, 20): KIT})
    return predictions, frames


def test_views_are_spread_over_time_not_taken_from_one_moment():
    predictions, frames = _clip(lambda frame: 1)
    config = JerseyViewConfig(views_per_identity=8, min_time_gap_ms=400, min_box_height=10.0)
    selected, crops = collect_identity_views(predictions, frames.get, {1: 1}, None, config)

    chosen = sorted(view.frame_index for view in selected[1])
    assert len(chosen) == 8
    assert len(crops) == 8
    gaps = [later - earlier for earlier, later in zip(chosen, chosen[1:], strict=False)]
    assert min(gaps) >= 10  # 400 ms at 25 fps

    # what the backend would have taken: the largest boxes, which in this clip
    # are eight consecutive frames around the moment the player is nearest
    largest = sorted(predictions, key=lambda detection: -detection["bbox_image"]["h"])[:8]
    naive = sorted(detection["frame"] for detection in largest)
    assert max(naive) - min(naive) == 7
    assert chosen != naive


def test_linked_tracklets_are_read_as_one_identity():
    predictions, frames = _clip(lambda frame: 1 if frame <= 100 else 2)
    config = JerseyViewConfig(views_per_identity=10, min_box_height=10.0)

    joined, _ = collect_identity_views(predictions, frames.get, {1: 1, 2: 1}, None, config)
    assert set(joined) == {1}
    assert {view.tracklet_id for view in joined[1]} == {"1", "2"}

    apart, _ = collect_identity_views(predictions, frames.get, {1: 1, 2: 2}, None, config)
    assert set(apart) == {1, 2}


def test_embeddings_suppress_near_duplicate_views():
    predictions, frames = _clip(lambda frame: 1)
    config = JerseyViewConfig(
        views_per_identity=8, min_time_gap_ms=0, min_box_height=10.0, max_cosine_similarity=0.9
    )
    identical = {(frame, 100.0, 20.0): np.array([1.0, 0.0]) for frame in range(1, 201)}
    selected, _ = collect_identity_views(predictions, frames.get, {1: 1}, identical, config)
    assert len(selected[1]) == 1  # every view looks the same, so one is enough

    rng = np.random.default_rng(7)
    varied = {(frame, 100.0, 20.0): rng.normal(size=32) for frame in range(1, 201)}
    spread, _ = collect_identity_views(predictions, frames.get, {1: 1}, varied, config)
    assert len(spread[1]) == 8


def test_small_boxes_are_never_offered_to_the_reader():
    predictions, frames = _clip(lambda frame: 1)
    for detection in predictions:
        detection["bbox_image"]["h"] = 30.0
    config = JerseyViewConfig(views_per_identity=8, min_box_height=40.0)
    selected, crops = collect_identity_views(predictions, frames.get, {1: 1}, None, config)
    assert selected == {}
    assert crops == {}


def test_manifest_lists_views_in_the_order_they_should_be_asked():
    predictions, frames = _clip(lambda frame: 1 if frame <= 100 else 2)
    config = JerseyViewConfig(views_per_identity=6, min_box_height=10.0)
    selected, _ = collect_identity_views(predictions, frames.get, {1: 1, 2: 1}, None, config)

    manifest = views_manifest(selected, "SNGS-000", config)
    assert manifest["sequence"] == "SNGS-000"
    entry = manifest["identities"][0]
    assert entry["identity"] == 1
    assert entry["tracklets"] == ["1", "2"]
    assert len(entry["views"]) == 6
    assert all(view["crop"].startswith("crops/") for view in entry["views"])
