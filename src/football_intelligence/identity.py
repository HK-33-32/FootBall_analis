"""Inspectable global tracklet-to-player identity solver."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .domain import GlobalPlayerMemory, TrackletMemory


@dataclass(frozen=True)
class IdentityConfig:
    reid_weight: float = 0.55
    team_weight: float = 0.20
    jersey_weight: float = 0.25
    merge_threshold: float = 0.70
    overlap_tolerance_ms: int = 200
    hard_team_confidence: float = 0.85
    hard_jersey_confidence: float = 0.90
    suppress_imbalanced_team_evidence: bool = True
    min_team_share: float = 0.20
    min_team_tracklets: int = 8


@dataclass(frozen=True)
class TeamEvidenceAudit:
    reliable: bool
    suppressed: bool
    tracklets_with_team: int
    shares: dict[str, float]
    reason: str | None = None


@dataclass(frozen=True)
class PairDecision:
    left_tracklet_id: str
    right_tracklet_id: str
    score: float
    merged: bool
    reasons: tuple[str, ...]


class GlobalIdentitySolver:
    def __init__(self, config: IdentityConfig | None = None):
        self.config = config or IdentityConfig()

    def solve(
        self, memories: list[TrackletMemory]
    ) -> tuple[list[GlobalPlayerMemory], list[PairDecision]]:
        audit = self.audit_team_evidence(memories)
        if audit.suppressed:
            memories = [item.model_copy(update={"team_posterior": {}}) for item in memories]
        ordered = sorted(memories, key=lambda item: item.tracklet_id)
        parent = {item.tracklet_id: item.tracklet_id for item in ordered}
        members = {item.tracklet_id: [item] for item in ordered}
        decisions: list[PairDecision] = []
        scored: list[tuple[float, TrackletMemory, TrackletMemory, list[str]]] = []
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                score, reasons = self._pair_score(left, right)
                scored.append((score, left, right, reasons))
        for score, left, right, reasons in sorted(scored, key=lambda row: -row[0]):
            root_left, root_right = (
                _find(parent, left.tracklet_id),
                _find(parent, right.tracklet_id),
            )
            merged = False
            if root_left != root_right and score >= self.config.merge_threshold:
                conflict = self._group_conflict(members[root_left], members[root_right])
                if conflict:
                    reasons = reasons + [conflict]
                else:
                    parent[root_right] = root_left
                    members[root_left].extend(members.pop(root_right))
                    merged = True
            decisions.append(
                PairDecision(left.tracklet_id, right.tracklet_id, score, merged, tuple(reasons))
            )
        players = [self._make_player(group) for _, group in sorted(members.items())]
        return players, decisions

    def audit_team_evidence(self, memories: list[TrackletMemory]) -> TeamEvidenceAudit:
        totals: dict[str, float] = {}
        labelled = 0
        for item in memories:
            if not item.team_posterior:
                continue
            labelled += 1
            for team, probability in item.team_posterior.items():
                totals[str(team)] = totals.get(str(team), 0.0) + probability
        denominator = sum(totals.values())
        shares = {
            team: value / denominator for team, value in sorted(totals.items())
        } if denominator else {}
        imbalanced = labelled >= self.config.min_team_tracklets and (
            len(shares) != 2 or min(shares.values()) < self.config.min_team_share
        )
        suppressed = self.config.suppress_imbalanced_team_evidence and imbalanced
        reason = "team_evidence_suppressed_global_imbalance" if suppressed else None
        return TeamEvidenceAudit(
            reliable=not imbalanced,
            suppressed=suppressed,
            tracklets_with_team=labelled,
            shares=shares,
            reason=reason,
        )

    def _pair_score(self, left: TrackletMemory, right: TrackletMemory) -> tuple[float, list[str]]:
        reasons: list[str] = []
        if _overlap(left, right) > self.config.overlap_tolerance_ms:
            return 0.0, ["temporal_overlap"]
        reid = _cosine(left.reid_embedding, right.reid_embedding)
        team = _distribution_similarity(left.team_posterior, right.team_posterior)
        jersey = _distribution_similarity(left.jersey_posterior, right.jersey_posterior)
        reasons.extend([f"reid={reid:.3f}", f"team={team:.3f}", f"jersey={jersey:.3f}"])
        has_reid_evidence = bool(left.reid_embedding and right.reid_embedding)
        has_jersey_evidence = bool(left.jersey_posterior and right.jersey_posterior)
        if not has_reid_evidence and not has_jersey_evidence:
            return 0.0, reasons + ["team_only_is_not_identity_evidence"]
        weighted = 0.0
        available = 0.0
        if has_reid_evidence:
            weighted += self.config.reid_weight * reid
            available += self.config.reid_weight
        if left.team_posterior and right.team_posterior:
            weighted += self.config.team_weight * team
            available += self.config.team_weight
        if has_jersey_evidence:
            weighted += self.config.jersey_weight * jersey
            available += self.config.jersey_weight
        return (weighted / available if available else 0.0, reasons)

    def _group_conflict(
        self, left: list[TrackletMemory], right: list[TrackletMemory]
    ) -> str | None:
        for one in left:
            for two in right:
                if _overlap(one, two) > self.config.overlap_tolerance_ms:
                    return "group_temporal_overlap"
        left_team, left_team_p = _aggregate_distribution(left, "team_posterior")
        right_team, right_team_p = _aggregate_distribution(right, "team_posterior")
        if (
            left_team
            and right_team
            and left_team != right_team
            and left_team_p >= self.config.hard_team_confidence
            and right_team_p >= self.config.hard_team_confidence
        ):
            return "high_confidence_team_conflict"
        left_jersey, left_jersey_p = _aggregate_distribution(left, "jersey_posterior")
        right_jersey, right_jersey_p = _aggregate_distribution(right, "jersey_posterior")
        if (
            left_jersey is not None
            and right_jersey is not None
            and left_jersey != right_jersey
            and left_jersey_p >= self.config.hard_jersey_confidence
            and right_jersey_p >= self.config.hard_jersey_confidence
        ):
            return "high_confidence_jersey_conflict"
        return None

    def _make_player(self, group: list[TrackletMemory]) -> GlobalPlayerMemory:
        group = sorted(group, key=lambda item: item.tracklet_id)
        team, team_confidence = _aggregate_distribution(group, "team_posterior")
        jersey, jersey_confidence = _aggregate_distribution(group, "jersey_posterior")
        digest = sha256("|".join(item.tracklet_id for item in group).encode()).hexdigest()[:12]
        evidence_strength = max(team_confidence, 0.5) * max(jersey_confidence, 0.5)
        return GlobalPlayerMemory(
            global_player_id=f"gp_{digest}",
            match_id=group[0].match_id,
            tracklet_ids=[item.tracklet_id for item in group],
            team=str(team) if team is not None else None,
            team_confidence=team_confidence,
            jersey_number=int(jersey) if jersey is not None else None,
            jersey_confidence=jersey_confidence,
            identity_confidence=min(1.0, evidence_strength * (0.8 + 0.05 * len(group))),
        )


def _find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _overlap(left: TrackletMemory, right: TrackletMemory) -> int:
    return max(
        0, min(left.last_seen_ms, right.last_seen_ms) - max(left.first_seen_ms, right.first_seen_ms)
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a, b = np.asarray(left), np.asarray(right)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, float(np.dot(a, b) / denominator)) if denominator else 0.0


def _distribution_similarity(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return float(sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys))


def _aggregate_distribution(group: list[TrackletMemory], field: str) -> tuple[object | None, float]:
    totals: dict[object, float] = {}
    for item in group:
        for key, value in getattr(item, field).items():
            totals[key] = totals.get(key, 0.0) + value
    if not totals:
        return None, 0.0
    best = max(totals, key=totals.get)
    denominator = sum(totals.values())
    return best, totals[best] / denominator if denominator else 0.0
