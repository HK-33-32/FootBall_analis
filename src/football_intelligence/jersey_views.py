"""Choosing which crops of a player to show a jersey reader.

The perception backend reads numbers per *tracklet*: it takes the eight largest
boxes of one fragment and asks the VLM about them. Two things go wrong with
that. The eight largest boxes of a fragment are usually consecutive frames of
one pose, so the model sees the same glimpse eight times; and a player broken
into three fragments is asked three separate questions, each from a third of
the evidence, which is how one fragment ends up calling a player 2 and another
calling the same player 22.

This module pools the crops of a whole stitched identity and picks views that
disagree with each other -- spread over time, and, when re-identification
embeddings are available, spread in appearance too. Selection itself is
delegated to :class:`~football_intelligence.player_memory.BestViewSelector`,
which is the component built for exactly this.

Player orientation is not estimated, so ``back_visibility`` and its siblings
are left at zero and those terms fall out of the ranking. Ranking therefore
rests on resolution, sharpness, occlusion and diversity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .domain import ViewObservation
from .gamestate import PLAYER_ROLES, Detection, FrameLoader, detection_key
from .player_memory import BestViewConfig, BestViewSelector


@dataclass(frozen=True)
class JerseyViewConfig:
    """How many crops to consider and how different they have to be."""

    candidates_per_identity: int = 60
    views_per_identity: int = 24
    min_box_height: float = 40.0
    max_crop_height: int = 320
    min_time_gap_ms: int = 400
    max_cosine_similarity: float = 0.96
    fps: float = 25.0


Box = tuple[float, float, float, float]


def _iou(left: Box, right: Box) -> float:
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    inner_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inner_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inner_w * inner_h
    if intersection <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def _box(detection: Detection) -> Box:
    box = detection["bbox_image"]
    x, y = float(box["x"]), float(box["y"])
    return x, y, x + float(box["w"]), y + float(box["h"])


def _occlusions(predictions: Sequence[Detection]) -> dict[int, float]:
    """Worst overlap of each detection with any other box in its frame."""
    by_frame: dict[int, list[int]] = {}
    for index, detection in enumerate(predictions):
        by_frame.setdefault(int(detection["frame"]), []).append(index)
    worst: dict[int, float] = {}
    for indices in by_frame.values():
        boxes = [_box(predictions[index]) for index in indices]
        for position, index in enumerate(indices):
            others = (
                _iou(boxes[position], boxes[other])
                for other in range(len(indices))
                if other != position
            )
            worst[index] = min(1.0, max(others, default=0.0))
    return worst


def collect_identity_views(
    predictions: Sequence[Detection],
    frames: FrameLoader,
    groups: dict[int, int],
    embeddings: dict[tuple[int, float, float], np.ndarray] | None = None,
    config: JerseyViewConfig | None = None,
) -> tuple[dict[int, list[ViewObservation]], dict[str, np.ndarray]]:
    """Return the chosen views per identity plus the crop behind each one.

    ``groups`` maps a track id to the id of the identity it belongs to, as
    produced by :func:`~football_intelligence.gamestate.link_tracklets`.
    ``embeddings`` are per-detection re-identification vectors keyed by the
    detection's ``(frame, box x, box y)``; when absent, views are spread over
    time only.
    """
    config = config or JerseyViewConfig()
    players = [
        (index, detection)
        for index, detection in enumerate(predictions)
        if (detection.get("attributes") or {}).get("role") in PLAYER_ROLES
        and float(detection["bbox_image"]["h"]) >= config.min_box_height
    ]
    occlusion = _occlusions([detection for _, detection in players])

    # Candidates are stratified over time before they are ranked. Taking the
    # largest boxes outright, as the backend does, hands back a run of adjacent
    # frames from whichever moment the player was nearest the camera -- one
    # glimpse repeated, which no amount of ranking afterwards can undo.
    bucket_frames = max(1, int(round(config.fps * config.min_time_gap_ms / 1000.0)))
    pools: dict[int, dict[int, tuple[int, Detection]]] = {}
    for position, (_, detection) in enumerate(players):
        identity = groups.get(detection["track_id"], detection["track_id"])
        bucket = int(detection["frame"]) // bucket_frames
        best = pools.setdefault(identity, {}).get(bucket)
        height = float(detection["bbox_image"]["h"])
        if best is None or height > float(best[1]["bbox_image"]["h"]):
            pools[identity][bucket] = (position, detection)

    candidates: dict[int, list[tuple[int, Detection]]] = {}
    for identity, buckets in pools.items():
        rows = sorted(buckets.values(), key=lambda row: -float(row[1]["bbox_image"]["h"]))
        candidates[identity] = rows[: config.candidates_per_identity]

    wanted: dict[int, list[tuple[int, Detection, int]]] = {}
    for identity, rows in candidates.items():
        for position, detection in rows:
            wanted.setdefault(int(detection["frame"]), []).append((position, detection, identity))

    views: dict[int, list[ViewObservation]] = {}
    crops: dict[str, np.ndarray] = {}
    for frame_number in sorted(wanted):
        image = frames(frame_number)
        if image is None:
            continue
        height, width = image.shape[:2]
        for position, detection, identity in wanted[frame_number]:
            x0, y0, x1, y1 = _box(detection)
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            if x1 - x0 < 4 or y1 - y0 < 8:
                continue
            crop = image[y0:y1, x0:x1]
            crop = _cap_height(crop, config.max_crop_height)
            observation_id = f"{frame_number}:{detection['track_id']}:{x0}:{y0}"
            crops[observation_id] = crop
            views.setdefault(identity, []).append(
                ViewObservation(
                    observation_id=observation_id,
                    tracklet_id=str(detection["track_id"]),
                    timestamp_ms=int(frame_number / config.fps * 1000),
                    frame_index=frame_number,
                    crop_uri=f"crops/{observation_id.replace(':', '_')}.png",
                    bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    sharpness=_sharpness(crop),
                    resolution_score=min(1.0, (y1 - y0) / config.max_crop_height),
                    occlusion=occlusion.get(position, 0.0),
                    back_visibility=0.0,
                    front_visibility=0.0,
                    side_visibility=0.0,
                    track_confidence=1.0,
                    embedding=_embedding(embeddings, frame_number, x0, y0),
                )
            )

    selector = BestViewSelector(
        BestViewConfig(
            top_k=config.views_per_identity,
            min_time_gap_ms=config.min_time_gap_ms,
            max_cosine_similarity=config.max_cosine_similarity,
        )
    )
    selected = {identity: selector.select(items) for identity, items in views.items()}
    keep = {view.observation_id for items in selected.values() for view in items}
    return selected, {key: value for key, value in crops.items() if key in keep}


def _cap_height(crop: np.ndarray, limit: int) -> np.ndarray:
    if crop.shape[0] <= limit:
        return crop
    import cv2

    scale = limit / crop.shape[0]
    return cv2.resize(
        crop, (max(4, int(crop.shape[1] * scale)), limit), interpolation=cv2.INTER_AREA
    )


def _sharpness(crop: np.ndarray) -> float:
    import cv2

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _embedding(
    embeddings: dict[tuple[int, float, float], np.ndarray] | None,
    frame: int,
    x: float,
    y: float,
) -> list[float]:
    if not embeddings:
        return []
    vector = embeddings.get(detection_key(frame, x, y))
    return list(map(float, vector)) if vector is not None else []


def views_manifest(
    selected: dict[int, list[ViewObservation]], sequence: str, config: JerseyViewConfig
) -> dict[str, Any]:
    """A reader-agnostic description of what to ask about, in ranked order."""
    return {
        "sequence": sequence,
        "views_per_identity": config.views_per_identity,
        "identities": [
            {
                "identity": identity,
                "tracklets": sorted({view.tracklet_id for view in items}),
                "views": [
                    {
                        "observation_id": view.observation_id,
                        "crop": view.crop_uri,
                        "frame": view.frame_index,
                        "height": view.bbox_xyxy[3] - view.bbox_xyxy[1],
                        "occlusion": round(view.occlusion, 4),
                        "sharpness": round(view.sharpness, 2),
                    }
                    for view in items
                ],
            }
            for identity, items in sorted(selected.items())
        ],
    }


__all__ = [
    "JerseyViewConfig",
    "collect_identity_views",
    "views_manifest",
]
