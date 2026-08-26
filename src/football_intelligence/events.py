"""Candidate conversion, evidence sampling, semantic inference, and active re-analysis."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2

from .cache import InferenceCache
from .domain import EvidenceRef, SemanticEvent
from .vlm import SemanticRequest, SemanticVLMBackend


@dataclass(frozen=True)
class ReanalysisConfig:
    event_confidence_threshold: float = 0.68
    alternative_margin_threshold: float = 0.18
    normal_before_ms: int = 3000
    normal_after_ms: int = 4000
    second_before_ms: int = 6000
    second_after_ms: int = 8000
    first_pass_fps: float = 1.5
    second_pass_fps: float = 4.0
    max_frames_first: int = 10
    max_frames_second: int = 24


class FrameSampler:
    def sample(
        self, video_path: str | Path, start_ms: int, end_ms: int, fps: float, max_frames: int
    ) -> tuple[list[int], list[str]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open source video: {video_path}")
        duration_ms = max(1, end_ms - start_ms)
        count = min(max_frames, max(1, int(duration_ms / 1000 * fps) + 1))
        timestamps = [
            round(start_ms + index * duration_ms / max(1, count - 1)) for index in range(count)
        ]
        images: list[str] = []
        accepted: list[int] = []
        try:
            for timestamp in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                scale = min(1.0, 960 / max(height, width))
                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (round(width * scale), round(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    images.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode())
                    accepted.append(timestamp)
        finally:
            capture.release()
        if not images:
            raise RuntimeError(f"no frames decoded from {video_path} in [{start_ms}, {end_ms}]")
        return accepted, images


class ActiveSemanticEngine:
    def __init__(
        self,
        first_pass: SemanticVLMBackend,
        second_pass: SemanticVLMBackend | None = None,
        config: ReanalysisConfig | None = None,
        sampler: FrameSampler | None = None,
        cache: InferenceCache | None = None,
    ):
        self.first_pass = first_pass
        self.second_pass = second_pass or first_pass
        self.config = config or ReanalysisConfig()
        self.sampler = sampler or FrameSampler()
        self.cache = cache

    def analyse_candidate(
        self,
        match_id: str,
        candidate_id: str,
        video_path: str | Path,
        centre_ms: int,
        candidate_hint: str,
        geometry: dict[str, Any],
        player_context: dict[str, Any],
        important: bool = False,
    ) -> SemanticEvent:
        first = self._run(
            backend=self.first_pass,
            match_id=match_id,
            candidate_id=candidate_id,
            video_path=video_path,
            centre_ms=centre_ms,
            before_ms=self.config.normal_before_ms,
            after_ms=self.config.normal_after_ms,
            fps=self.config.first_pass_fps,
            max_frames=self.config.max_frames_first,
            candidate_hint=candidate_hint,
            geometry=geometry,
            player_context=player_context,
            pass_number=1,
            trigger_reasons=[],
        )
        reasons = self.trigger_reasons(first, important)
        if not reasons:
            return first
        return self._run(
            backend=self.second_pass,
            match_id=match_id,
            candidate_id=candidate_id,
            video_path=video_path,
            centre_ms=centre_ms,
            before_ms=self.config.second_before_ms,
            after_ms=self.config.second_after_ms,
            fps=self.config.second_pass_fps,
            max_frames=self.config.max_frames_second,
            candidate_hint=candidate_hint,
            geometry=geometry,
            player_context=player_context,
            pass_number=2,
            trigger_reasons=reasons,
        )

    def trigger_reasons(self, event: SemanticEvent, important: bool = False) -> list[str]:
        reasons: list[str] = []
        if event.confidence < self.config.event_confidence_threshold:
            reasons.append("low_event_confidence")
        if event.insufficient_evidence:
            reasons.append("insufficient_evidence")
        alternatives = sorted((item.confidence for item in event.alternatives), reverse=True)
        if (
            alternatives
            and event.confidence - alternatives[0] < self.config.alternative_margin_threshold
        ):
            reasons.append("small_top1_top2_margin")
        if important:
            reasons.append("important_event")
        return reasons

    def _run(self, **kwargs: Any) -> SemanticEvent:
        backend: SemanticVLMBackend = kwargs["backend"]
        centre_ms: int = kwargs["centre_ms"]
        start_ms = max(0, centre_ms - kwargs["before_ms"])
        end_ms = centre_ms + kwargs["after_ms"]
        timestamps, images = self.sampler.sample(
            kwargs["video_path"], start_ms, end_ms, kwargs["fps"], kwargs["max_frames"]
        )
        request = SemanticRequest(
            match_id=kwargs["match_id"],
            start_ms=start_ms,
            end_ms=end_ms,
            timestamps_ms=timestamps,
            images_data_uri=images,
            geometry=kwargs["geometry"],
            player_context=kwargs["player_context"],
            candidate_hint=kwargs["candidate_hint"],
        )
        source = Path(kwargs["video_path"])
        stat = source.stat()
        source_digest = _file_sha256(str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        cache_payload = {
            "video_sha256": source_digest,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "sampling": {"fps": kwargs["fps"], "max_frames": kwargs["max_frames"]},
            "timestamps_ms": timestamps,
            "image_sha256": [sha256(image.encode()).hexdigest() for image in images],
            "model_name": backend.model_name,
            "model_revision": backend.model_revision,
            "prompt_version": backend.prompt_version,
            "schema_version": "1.0.0",
            "generation_config": backend.generation_config,
            "candidate_hint": kwargs["candidate_hint"],
            "geometry": kwargs["geometry"],
            "player_context": kwargs["player_context"],
        }
        cache_key = InferenceCache.key(cache_payload)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is None:
            output = backend.classify(request)
            if self.cache:
                self.cache.put(cache_key, output.model_dump(mode="json"))
        else:
            from .vlm.backend import VLMOutput

            output = VLMOutput.model_validate(cached)
        evidence_id = (
            "evd_"
            + sha256(f"{source.resolve()}:{start_ms}:{end_ms}:{timestamps}".encode()).hexdigest()[
                :16
            ]
        )
        evidence = EvidenceRef(
            evidence_id=evidence_id,
            match_id=kwargs["match_id"],
            source_uri=str(source.resolve()),
            start_ms=start_ms,
            end_ms=end_ms,
            timestamps_ms=timestamps,
            source_kind="video",
            content_sha256=source_digest,
        )
        actor = output.actor_tracklet_id
        target = output.target_tracklet_id
        tracklet_map = kwargs["player_context"].get("tracklet_to_global_player", {})
        return SemanticEvent(
            event_id=f"sem_{kwargs['candidate_id']}_p{kwargs['pass_number']}",
            match_id=kwargs["match_id"],
            primary_event=output.primary_event,
            confidence=output.confidence,
            actor_global_player_id=tracklet_map.get(actor, actor),
            target_global_player_id=tracklet_map.get(target, target),
            outcome=output.outcome,
            start_ms=start_ms,
            end_ms=end_ms,
            evidence=[evidence],
            alternatives=output.alternatives,
            insufficient_evidence=output.insufficient_evidence,
            analysis_pass=kwargs["pass_number"],
            trigger_reasons=kwargs["trigger_reasons"],
            model_name=backend.model_name,
            model_revision=backend.model_revision,
            prompt_version=backend.prompt_version,
            raw_model_score=output.confidence,
            attributes={"candidate_hint": kwargs["candidate_hint"], "geometry": kwargs["geometry"]},
        )


@lru_cache(maxsize=16)
def _file_sha256(path: str, _size: int, _mtime_ns: int) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


LEGACY_CANDIDATE_TYPES = {"pass", "shot", "challenge", "interception", "recovery", "touch"}


def legacy_event_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert geometric ledger entries into candidates, never final semantic labels."""
    output: list[dict[str, Any]] = []
    for index, event in enumerate(payload.get("events", [])):
        kind = str(event.get("type", "")).lower()
        if kind not in LEGACY_CANDIDATE_TYPES:
            continue
        clock = event.get("clock", {})
        centre_ms = round(float(clock.get("video_time_s", 0)) * 1000)
        output.append(
            {
                "candidate_id": str(event.get("event_id", f"legacy_{index:06d}")),
                "centre_ms": centre_ms,
                "candidate_hint": kind,
                "geometry": {
                    "start_xy": event.get("start_xy"),
                    "end_xy": event.get("end_xy"),
                    "attributes": event.get("attributes", {}),
                    "legacy_confidence": event.get("confidence"),
                },
                "player_context": {
                    "actor_tracklet_id": event.get("track_id"),
                    "target_tracklet_id": event.get("related_track_id"),
                    "team": event.get("team"),
                },
                "important": kind == "shot",
            }
        )
    return output
