"""Break a GS-HOTA score down into detection, role, team and jersey accuracy.

GS-HOTA collapses geometry and identity into one number, which makes it a poor
debugging tool: a perfect tracker with a broken team classifier and a broken
tracker with perfect attributes can score the same. This script matches
predictions to ground truth on pitch geometry alone -- the same Gaussian
similarity and 5 m tolerance the official evaluator uses -- and then reports
each attribute's accuracy on the matched pairs.

It reads ground truth, so it is a diagnostic, never part of the pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_FLOOR = 0.05  # similarity at exactly the distance tolerance


def gaussian_sigma(tolerance_m: float) -> float:
    return tolerance_m / float(np.sqrt(-2 * np.log(MATCH_FLOOR)))


def _centre(annotation: dict[str, Any]) -> tuple[float, float]:
    pitch = annotation["bbox_pitch"]
    return pitch["x_bottom_middle"], pitch["y_bottom_middle"]


def _by_frame(annotations: list[dict[str, Any]], frame_of) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if not annotation.get("bbox_pitch"):
            continue
        if (annotation.get("attributes") or {}).get("role") == "ball":
            continue
        grouped[frame_of(annotation)].append(annotation)
    return grouped


def report(sequence_dir: Path, predictions_path: Path, tolerance_m: float) -> dict[str, Any]:
    truth = json.loads((sequence_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_index = {image["image_id"]: int(str(image["image_id"])[-6:]) for image in truth["images"]}
    ground_truth = _by_frame(truth["annotations"], lambda a: frame_index[a["image_id"]])
    predicted = _by_frame(
        json.loads(predictions_path.read_text(encoding="utf-8"))["predictions"],
        lambda a: int(a["frame"]),
    )

    sigma = gaussian_sigma(tolerance_m)
    counts: Counter = Counter()
    role_hits = role_total = 0
    team_hits = team_total = 0
    jersey_exact = jersey_total = 0
    jersey_labelled = jersey_emitted = jersey_correct = jersey_false_positive = 0
    for frame in sorted(ground_truth):
        expected, actual = ground_truth[frame], predicted.get(frame, [])
        if not expected or not actual:
            counts["missed"] += len(expected)
            counts["spurious"] += len(actual)
            continue
        distances = np.linalg.norm(
            np.array([_centre(a) for a in expected])[:, None, :]
            - np.array([_centre(a) for a in actual])[None, :, :],
            axis=2,
        )
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        rows, columns = linear_sum_assignment(-similarity)
        matched_rows, matched_columns = set(), set()
        for row, column in zip(rows, columns, strict=True):
            if similarity[row, column] < MATCH_FLOOR:
                continue
            matched_rows.add(row)
            matched_columns.add(column)
            counts["matched"] += 1
            want, got = expected[row]["attributes"], actual[column]["attributes"]
            role_total += 1
            role_hits += want["role"] == got.get("role")
            if want["role"] in ("player", "goalkeeper"):
                team_total += 1
                team_hits += want["team"] == got.get("team")
            if want["role"] == "player":
                jersey_total += 1
                want_number = None if want["jersey"] is None else str(want["jersey"])
                got_number = got.get("jersey")
                got_number = None if got_number in (None, "") else str(got_number)
                jersey_exact += want_number == got_number
                if want_number is not None:
                    jersey_labelled += 1
                    if got_number is not None:
                        jersey_emitted += 1
                        jersey_correct += want_number == got_number
                elif got_number is not None:
                    jersey_false_positive += 1
        counts["missed"] += len(expected) - len(matched_rows)
        counts["spurious"] += len(actual) - len(matched_columns)

    matched = counts["matched"]
    return {
        "sequence": sequence_dir.name,
        "predictions": str(predictions_path),
        "distance_tolerance_m": tolerance_m,
        "detection": {
            "matched": matched,
            "missed": counts["missed"],
            "spurious": counts["spurious"],
            "recall": round(matched / max(matched + counts["missed"], 1), 4),
            "precision": round(matched / max(matched + counts["spurious"], 1), 4),
        },
        "role_accuracy": round(role_hits / max(role_total, 1), 4),
        "team_accuracy": round(team_hits / max(team_total, 1), 4),
        "jersey": {
            "exact_including_abstention": round(jersey_exact / max(jersey_total, 1), 4),
            "accuracy_when_emitted": round(jersey_correct / max(jersey_emitted, 1), 4),
            "coverage_of_labelled": round(jersey_emitted / max(jersey_labelled, 1), 4),
            "emitted_where_truth_has_none": jersey_false_positive,
            "labelled_detections": jersey_labelled,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--soccernet-root", type=Path, default=Path("data/soccernet/valid"))
    parser.add_argument("--distance-tolerance-m", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = report(
        args.soccernet_root / args.sequence, args.predictions, args.distance_tolerance_m
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
