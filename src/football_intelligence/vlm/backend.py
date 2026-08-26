"""Replaceable structured semantic VLM backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..domain import CandidateScore, EventOutcome, EventType

PROMPT_VERSION = "event-v1.0.0"


class VLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_event: EventType
    confidence: float = Field(ge=0, le=1)
    actor_tracklet_id: str | None = None
    target_tracklet_id: str | None = None
    outcome: EventOutcome = EventOutcome.UNKNOWN
    alternatives: list[CandidateScore] = Field(default_factory=list)
    insufficient_evidence: bool = False


@dataclass(frozen=True)
class SemanticRequest:
    match_id: str
    start_ms: int
    end_ms: int
    timestamps_ms: list[int]
    images_data_uri: list[str]
    geometry: dict[str, Any] = field(default_factory=dict)
    player_context: dict[str, Any] = field(default_factory=dict)
    candidate_hint: str | None = None


class SemanticVLMBackend(ABC):
    model_name: str
    model_revision: str
    prompt_version: str = PROMPT_VERSION

    @property
    def generation_config(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def classify(self, request: SemanticRequest) -> VLMOutput:
        raise TypeError("SemanticVLMBackend.classify cannot be called on the abstract base")


class OpenAICompatibleVLM(SemanticVLMBackend):
    """Strict JSON-schema calls to a local vLLM/SGLang/OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        model_revision: str = "a115a837cf3cbba4b697aa74b609721b5009ed41",
        api_key: str = "EMPTY",
        timeout_s: float = 300.0,
        max_tokens: int = 384,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.model_revision = model_revision
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._prompt = (Path(__file__).parent / "prompts" / "event_v1.txt").read_text(
            encoding="utf-8"
        )

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": "json_schema",
        }

    def classify(self, request: SemanticRequest) -> VLMOutput:
        content: list[dict[str, Any]] = []
        for timestamp_ms, image_uri in zip(
            request.timestamps_ms, request.images_data_uri, strict=True
        ):
            content.append({"type": "text", "text": f"frame_timestamp_ms={timestamp_ms}"})
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        context = {
            "match_id": request.match_id,
            "window": [request.start_ms, request.end_ms],
            "candidate_hint": request.candidate_hint,
            "geometry": request.geometry,
            "player_context": request.player_context,
        }
        content.append(
            {
                "type": "text",
                "text": self._prompt + "\nCONTEXT=" + json.dumps(context, separators=(",", ":")),
            }
        )
        schema = VLMOutput.model_json_schema()
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "semantic_event", "strict": True, "schema": schema},
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            return VLMOutput.model_validate_json(raw)
        except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            identity = f"{self.model_name}@{self.model_revision}"
            raise RuntimeError(
                f"semantic VLM failed validation at {self.base_url}; model={identity}: {exc}"
            ) from exc
