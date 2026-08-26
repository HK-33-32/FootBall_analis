"""Versioned domain contracts shared by inference, storage, API, and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventType(StrEnum):
    TOUCH = "touch"
    CONTROLLED_POSSESSION = "controlled_possession"
    INTENTIONAL_PASS = "intentional_pass"
    CLEARANCE = "clearance"
    INTERCEPTION = "interception"
    RECOVERY = "recovery"
    TACKLE = "tackle"
    DUEL = "duel"
    SHOT = "shot"
    UNKNOWN = "unknown"


class EventOutcome(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    WON = "won"
    LOST = "lost"
    UNKNOWN = "unknown"


class EvidenceRef(StrictModel):
    evidence_id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    timestamps_ms: list[int] = Field(default_factory=list)
    frame_indices: list[int] = Field(default_factory=list)
    crop_boxes_xyxy: list[tuple[float, float, float, float]] = Field(default_factory=list)
    source_kind: str = "video"
    content_sha256: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> EvidenceRef:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if any(t < self.start_ms or t > self.end_ms for t in self.timestamps_ms):
            raise ValueError("evidence timestamp outside interval")
        return self


class CandidateScore(StrictModel):
    label: EventType
    confidence: float = Field(ge=0, le=1)


class SemanticEvent(StrictModel):
    schema_version: str = "1.0.0"
    event_id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    primary_event: EventType
    confidence: float = Field(ge=0, le=1)
    actor_global_player_id: str | None = None
    target_global_player_id: str | None = None
    outcome: EventOutcome = EventOutcome.UNKNOWN
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    evidence: list[EvidenceRef] = Field(min_length=1)
    alternatives: list[CandidateScore] = Field(default_factory=list)
    insufficient_evidence: bool = False
    analysis_pass: int = Field(default=1, ge=1, le=2)
    trigger_reasons: list[str] = Field(default_factory=list)
    model_name: str
    model_revision: str
    prompt_version: str
    raw_model_score: float | None = Field(default=None, ge=0, le=1)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event(self) -> SemanticEvent:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if self.insufficient_evidence and self.primary_event != EventType.UNKNOWN:
            raise ValueError("insufficient evidence must abstain with primary_event=unknown")
        return self


class JerseyEvidence(StrictModel):
    number: int | None = Field(default=None, ge=0, le=99)
    confidence: float = Field(ge=0, le=1)
    observation_id: str
    model_name: str
    model_revision: str
    prompt_version: str


class ViewObservation(StrictModel):
    observation_id: str
    tracklet_id: str
    timestamp_ms: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    crop_uri: str
    bbox_xyxy: tuple[float, float, float, float]
    sharpness: float = Field(ge=0)
    resolution_score: float = Field(ge=0, le=1)
    occlusion: float = Field(ge=0, le=1)
    back_visibility: float = Field(ge=0, le=1)
    front_visibility: float = Field(ge=0, le=1)
    side_visibility: float = Field(ge=0, le=1)
    track_confidence: float = Field(ge=0, le=1)
    embedding: list[float] = Field(default_factory=list)


class TrackletMemory(StrictModel):
    tracklet_id: str
    match_id: str
    team_posterior: dict[str, float] = Field(default_factory=dict)
    jersey_posterior: dict[int, float] = Field(default_factory=dict)
    role_posterior: dict[str, float] = Field(default_factory=dict)
    reid_embedding: list[float] = Field(default_factory=list)
    first_seen_ms: int = Field(ge=0)
    last_seen_ms: int = Field(ge=0)
    best_views: list[ViewObservation] = Field(default_factory=list)
    jersey_evidence: list[JerseyEvidence] = Field(default_factory=list)


class GlobalPlayerMemory(StrictModel):
    global_player_id: str
    match_id: str
    tracklet_ids: list[str] = Field(min_length=1)
    team: str | None = None
    team_confidence: float = Field(default=0, ge=0, le=1)
    jersey_number: int | None = Field(default=None, ge=0, le=99)
    jersey_confidence: float = Field(default=0, ge=0, le=1)
    roster_player_id: str | None = None
    roster_name: str | None = None
    identity_confidence: float = Field(default=0, ge=0, le=1)
    unresolved_conflicts: list[str] = Field(default_factory=list)


class ModelRun(StrictModel):
    schema_version: str = "1.0.0"
    run_id: str
    timestamp: str
    git_commit: str
    dirty_worktree: bool
    dataset: str
    split: str
    config_hash: str
    models: dict[str, dict[str, Any]]
    prompt_versions: dict[str, str]
    schema_versions: dict[str, str]
    hardware: dict[str, Any]
    software: dict[str, Any]
    seed: int = 0
