"""Deciding which frames the pitch calibration can be trusted on.

Positions come from a per-frame homography estimated from the pitch markings.
When the broadcast cuts to a camera the estimator was not trained for, it does
not fail loudly -- it returns a homography, and every player gets a position
that is simply wrong. Measured on broadcast footage of the 2022 final, after
one such cut the mapping was *mirrored*: the rank correlation between a
player's position across the image and his position along the pitch went from
+0.87 before the cut to -0.97 after it, so the left winger appeared on the
right touchline. Another frame collapsed twelve players onto a single point.

Statistics computed over those frames are not noisy, they are meaningless, and
the honest thing is to mark the frames as uncalibrated rather than to publish
positions derived from them. Three tests, all free of ground truth:

* **order** -- a camera looking at a plane maps left-to-right in the image onto
  a monotone order along some pitch axis. A frame whose order disagrees with
  the rest of the clip is mirrored, and one whose order is only loosely
  monotone is a homography that has started to drift. The floor is set where
  the healthy part of that footage never went: before the cut no frame fell
  below a rank correlation of 0.7, while a fifth of the frames that survived
  the mirror test after it did;
* **collapse** -- a dozen people on a pitch cannot occupy a few metres;
* **containment** -- players standing on the pitch cannot project outside it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

Detection = dict[str, Any]


@dataclass(frozen=True)
class CalibrationConfig:
    """What a frame has to satisfy for its positions to be published."""

    min_people: int = 6
    min_axis_correlation: float = 0.7
    min_pitch_extent_m: float = 8.0
    max_outside_share: float = 0.25
    pitch_half_length_m: float = 52.5
    pitch_half_width_m: float = 34.0
    out_of_pitch_margin_m: float = 3.0


def _frame_geometry(rows: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Axis correlation, pitch extent and out-of-pitch share for one frame."""
    image_x = np.array([row[0] for row in rows], dtype=float)
    pitch_x = np.array([row[1] for row in rows], dtype=float)
    pitch_y = np.array([row[2] for row in rows], dtype=float)
    def rank_correlation(values: np.ndarray) -> float:
        # a constant axis has no defined correlation, and scipy warns about it
        if values.min() == values.max() or image_x.min() == image_x.max():
            return 0.0
        result = spearmanr(image_x, values).correlation
        return 0.0 if np.isnan(result) else float(result)

    along = rank_correlation(pitch_x)
    across = rank_correlation(pitch_y)
    # whichever pitch axis the image order tracks -- a camera behind the goal
    # tracks the other one, and both are legitimate
    correlation = along if abs(along) >= abs(across) else across
    extent = float(np.hypot(pitch_x.max() - pitch_x.min(), pitch_y.max() - pitch_y.min()))
    return correlation, extent, 0.0


def frame_validity(
    predictions: Sequence[Detection], config: CalibrationConfig | None = None
) -> dict[str, Any]:
    """Classify every frame as calibrated or not, with the reason."""
    config = config or CalibrationConfig()
    by_frame: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for detection in predictions:
        attributes = detection.get("attributes") or {}
        pitch = detection.get("bbox_pitch") or {}
        if attributes.get("role") == "ball" or "x_bottom_middle" not in pitch:
            continue
        box = detection["bbox_image"]
        by_frame[int(detection["frame"])].append(
            (
                float(box["x"]) + float(box["w"]) / 2,
                float(pitch["x_bottom_middle"]),
                float(pitch["y_bottom_middle"]),
            )
        )

    limit_x = config.pitch_half_length_m + config.out_of_pitch_margin_m
    limit_y = config.pitch_half_width_m + config.out_of_pitch_margin_m

    measured: dict[int, dict[str, float]] = {}
    for frame, rows in by_frame.items():
        outside = sum(1 for _, x, y in rows if abs(x) > limit_x or abs(y) > limit_y)
        if len(rows) < config.min_people:
            measured[frame] = {"people": len(rows), "outside": outside / len(rows)}
            continue
        correlation, extent, _ = _frame_geometry(rows)
        measured[frame] = {
            "people": len(rows),
            "correlation": correlation,
            "extent": extent,
            "outside": outside / len(rows),
        }

    # The clip's own dominant orientation is the reference: a frame is mirrored
    # relative to the footage it belongs to, not to an absolute convention.
    confident = [
        entry["correlation"]
        for entry in measured.values()
        if abs(entry.get("correlation", 0.0)) >= 0.8
    ]
    reference = float(np.sign(np.median(confident))) if confident else 0.0

    rejected: dict[int, str] = {}
    for frame, entry in measured.items():
        if entry["outside"] > config.max_outside_share:
            rejected[frame] = "players projected off the pitch"
            continue
        if "correlation" not in entry:
            continue  # too few people to judge; left alone
        if entry["extent"] < config.min_pitch_extent_m:
            rejected[frame] = "everyone collapsed onto one spot"
            continue
        if abs(entry["correlation"]) < config.min_axis_correlation:
            rejected[frame] = "image order does not follow any pitch axis"
            continue
        if reference and np.sign(entry["correlation"]) != reference:
            rejected[frame] = "pitch mapping is mirrored"

    counts: dict[str, int] = defaultdict(int)
    for reason in rejected.values():
        counts[reason] += 1
    return {
        "reference_orientation": reference,
        "frames_examined": len(measured),
        "frames_rejected": len(rejected),
        "rejected": rejected,
        "reasons": dict(counts),
    }


def drop_uncalibrated(
    predictions: Sequence[Detection], config: CalibrationConfig | None = None
) -> tuple[list[Detection], dict[str, Any]]:
    """Return the detections on frames whose calibration survived the tests."""
    config = config or CalibrationConfig()
    report = frame_validity(predictions, config)
    rejected = set(report["rejected"])
    limit_x = config.pitch_half_length_m + config.out_of_pitch_margin_m
    limit_y = config.pitch_half_width_m + config.out_of_pitch_margin_m

    kept: list[Detection] = []
    off_pitch = 0
    for detection in predictions:
        if int(detection["frame"]) in rejected:
            continue
        pitch = detection.get("bbox_pitch") or {}
        # a frame can pass the tests above and still carry one impossible point
        if "x_bottom_middle" in pitch and (
            abs(float(pitch["x_bottom_middle"])) > limit_x
            or abs(float(pitch["y_bottom_middle"])) > limit_y
        ):
            off_pitch += 1
            continue
        kept.append(detection)

    report["detections_in"] = len(predictions)
    report["detections_kept"] = len(kept)
    report["detections_off_pitch"] = off_pitch
    return kept, report


__all__ = ["CalibrationConfig", "drop_uncalibrated", "frame_validity"]
