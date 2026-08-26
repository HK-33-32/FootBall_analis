"""Deterministic semantic classification and calibration metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def semantic_metrics(rows: Iterable[dict], labels: list[str] | None = None, bins: int = 10) -> dict:
    records = list(rows)
    labels = labels or sorted(
        {str(row["truth"]) for row in records} | {str(row["prediction"]) for row in records}
    )
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    abstained = 0
    confidences: list[float] = []
    correct: list[int] = []
    for row in records:
        truth, prediction = str(row["truth"]), str(row["prediction"])
        if row.get("insufficient_evidence", False):
            abstained += 1
            continue
        if truth not in index or prediction not in index:
            raise ValueError(f"unknown label in evaluation row: {truth=}, {prediction=}")
        matrix[index[truth], index[prediction]] += 1
        confidences.append(float(row["confidence"]))
        correct.append(int(truth == prediction))
    per_class = {}
    f1s = []
    for label, idx in index.items():
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[idx].sum()),
        }
        f1s.append(f1)
    return {
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "coverage": (len(records) - abstained) / len(records) if records else 0.0,
        "abstained": abstained,
        "ece": _ece(confidences, correct, bins),
        "sample_size": len(records),
    }


def _ece(confidences: list[float], correct: list[int], bins: int) -> float:
    if not confidences:
        return 0.0
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for confidence, hit in zip(confidences, correct, strict=True):
        grouped[min(bins - 1, int(max(0.0, min(1.0, confidence)) * bins))].append((confidence, hit))
    total = len(confidences)
    return sum(
        len(items)
        / total
        * abs(
            sum(item[0] for item in items) / len(items)
            - sum(item[1] for item in items) / len(items)
        )
        for items in grouped.values()
    )
