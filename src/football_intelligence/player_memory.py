"""Best-view mining and calibrated multi-view evidence aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .domain import JerseyEvidence, TrackletMemory, ViewObservation


@dataclass(frozen=True)
class BestViewConfig:
    top_k: int = 8
    min_time_gap_ms: int = 500
    max_cosine_similarity: float = 0.96
    sharpness_scale: float = 250.0
    weights: tuple[float, ...] = (0.18, 0.18, 0.16, 0.24, 0.08, 0.04, 0.12)


class BestViewSelector:
    def __init__(self, config: BestViewConfig | None = None):
        self.config = config or BestViewConfig()

    def score(self, view: ViewObservation) -> float:
        sharpness = 1.0 - math.exp(-view.sharpness / self.config.sharpness_scale)
        visibility = max(
            view.back_visibility, 0.55 * view.front_visibility, 0.35 * view.side_visibility
        )
        features = (
            sharpness,
            view.resolution_score,
            1.0 - view.occlusion,
            visibility,
            view.back_visibility,
            view.side_visibility,
            view.track_confidence,
        )
        return float(sum(a * b for a, b in zip(self.config.weights, features, strict=True)))

    def select(self, views: list[ViewObservation]) -> list[ViewObservation]:
        ranked = sorted(
            views, key=lambda view: (-self.score(view), view.timestamp_ms, view.observation_id)
        )
        selected: list[ViewObservation] = []
        for candidate in ranked:
            if any(
                abs(candidate.timestamp_ms - kept.timestamp_ms) < self.config.min_time_gap_ms
                for kept in selected
            ):
                continue
            if candidate.embedding and any(
                _cosine(candidate.embedding, kept.embedding) > self.config.max_cosine_similarity
                for kept in selected
                if kept.embedding
            ):
                continue
            selected.append(candidate)
            if len(selected) == self.config.top_k:
                break
        return selected


def aggregate_jersey_posterior(
    evidence: list[JerseyEvidence],
    allowed_numbers: set[int] | None = None,
    prior_strength: float = 0.2,
) -> dict[int, float]:
    """Log-opinion pooling with abstentions and per-observation de-duplication."""
    best_by_observation: dict[tuple[str, int], JerseyEvidence] = {}
    for item in evidence:
        if item.number is None or (
            allowed_numbers is not None and item.number not in allowed_numbers
        ):
            continue
        key = (item.observation_id, item.number)
        if key not in best_by_observation or item.confidence > best_by_observation[key].confidence:
            best_by_observation[key] = item
    numbers = sorted(
        allowed_numbers
        or {item.number for item in best_by_observation.values() if item.number is not None}
    )
    if not numbers:
        return {}
    logits = {number: math.log(prior_strength / len(numbers)) for number in numbers}
    grouped: dict[str, list[JerseyEvidence]] = defaultdict(list)
    for item in best_by_observation.values():
        grouped[item.observation_id].append(item)
    for alternatives in grouped.values():
        total = sum(max(item.confidence, 1e-6) for item in alternatives)
        for item in alternatives:
            assert item.number is not None
            logits[item.number] += math.log1p(9.0 * item.confidence / total)
    offset = max(logits.values())
    unnormalised = {number: math.exp(value - offset) for number, value in logits.items()}
    denominator = sum(unnormalised.values())
    return {number: value / denominator for number, value in unnormalised.items()}


def update_tracklet_memory(
    memory: TrackletMemory,
    views: list[ViewObservation],
    evidence: list[JerseyEvidence],
    selector: BestViewSelector,
    allowed_numbers: set[int] | None = None,
) -> TrackletMemory:
    return memory.model_copy(
        update={
            "best_views": selector.select(views),
            "jersey_evidence": evidence,
            "jersey_posterior": aggregate_jersey_posterior(evidence, allowed_numbers),
        }
    )


def _cosine(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0
