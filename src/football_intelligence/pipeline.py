"""End-to-end orchestration from specialist perception to semantic Match Memory."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import ModelRun, TrackletMemory
from .events import ActiveSemanticEngine, legacy_event_candidates
from .identity import GlobalIdentitySolver
from .memory import MatchMemory
from .perception import LegacyCoreClient


class MatchPipeline:
    def __init__(
        self,
        memory: MatchMemory,
        semantic_engine: ActiveSemanticEngine | None,
        perception: LegacyCoreClient | None = None,
        identity_solver: GlobalIdentitySolver | None = None,
    ):
        self.memory = memory
        self.semantic_engine = semantic_engine
        self.perception = perception
        self.identity_solver = identity_solver or GlobalIdentitySolver()

    def run(
        self,
        match_id: str,
        video_path: str | Path,
        start_s: float,
        end_s: float,
        fps: int = 12,
        legacy_events: dict[str, Any] | None = None,
        predictions: dict[str, Any] | None = None,
        roster: dict | None = None,
        render_video: bool = True,
        run_semantics: bool = True,
        dataset: str = "user-media",
        split: str = "unlabelled",
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        source = Path(video_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        notify = progress or (lambda _stage, _value: None)
        notify("perception", 0.02)
        legacy_job: dict[str, Any] | None = None
        if legacy_events is None or predictions is None:
            if self.perception is None:
                raise RuntimeError(
                    "perception artifacts were not supplied and FI_PERCEPTION_URL is not configured"
                )
            bundle = self.perception.analyse_video(
                source,
                start_s,
                end_s,
                fps=fps,
                roster=roster,
                render_video=render_video,
                progress_callback=lambda job: notify(
                    "perception", 0.05 + 0.45 * float(job.get("progress", 0))
                ),
            )
            legacy_job = bundle["job"]
            legacy_events, predictions = bundle["events"], bundle["predictions"]
        notify("identity", 0.55)
        tracklets = build_tracklet_memories(match_id, predictions or {}, fps=fps, start_s=start_s)
        team_evidence = self.identity_solver.audit_team_evidence(tracklets)
        players, pair_decisions = self.identity_solver.solve(tracklets)
        self.memory.put_players(players)
        tracklet_to_global = {
            tracklet_id: player.global_player_id
            for player in players
            for tracklet_id in player.tracklet_ids
        }
        candidates = legacy_event_candidates(legacy_events or {})
        semantic_events = []
        if run_semantics:
            if self.semantic_engine is None:
                raise RuntimeError(
                    "semantic inference requested but FI_VLM_BASE_URL is not configured"
                )
            total = max(1, len(candidates))
            for index, candidate in enumerate(candidates):
                context = dict(candidate["player_context"])
                context["tracklet_to_global_player"] = tracklet_to_global
                semantic_events.append(
                    self.semantic_engine.analyse_candidate(
                        match_id=match_id,
                        candidate_id=candidate["candidate_id"],
                        video_path=source,
                        centre_ms=candidate["centre_ms"],
                        candidate_hint=candidate["candidate_hint"],
                        geometry=candidate["geometry"],
                        player_context=context,
                        important=candidate["important"],
                    )
                )
                notify("semantics", 0.60 + 0.35 * (index + 1) / total)
            self.memory.put_events(semantic_events)
        config = {
            "fps": fps,
            "start_s": start_s,
            "end_s": end_s,
            "run_semantics": run_semantics,
            "candidate_count": len(candidates),
        }
        run = make_run_metadata(
            dataset=dataset,
            split=split,
            config=config,
            models={
                "perception": {"name": "football-core", "revision": "external-measured-baseline"},
                "semantic": {
                    "name": self.semantic_engine.first_pass.model_name
                    if self.semantic_engine
                    else "disabled",
                    "revision": self.semantic_engine.first_pass.model_revision
                    if self.semantic_engine
                    else "disabled",
                },
            },
        )
        self.memory.put_run(run)
        notify("done", 1.0)
        return {
            "match_id": match_id,
            "run": run.model_dump(mode="json"),
            "legacy_job": legacy_job,
            "legacy_statistics": bundle["players"] if legacy_job else None,
            "tracklets": len(tracklets),
            "team_evidence": team_evidence.__dict__,
            "global_players": [player.model_dump(mode="json") for player in players],
            "pair_decisions": [decision.__dict__ for decision in pair_decisions],
            "candidates": candidates,
            "semantic_events": [event.model_dump(mode="json") for event in semantic_events],
            "statistics": self.memory.statistics(match_id),
        }


def build_tracklet_memories(
    match_id: str, payload: dict[str, Any], fps: float, start_s: float
) -> list[TrackletMemory]:
    rows = payload.get("predictions", payload if isinstance(payload, list) else [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        role = str(row.get("attributes", {}).get("role", "")).lower()
        if role not in {"player", "goalkeeper"}:
            continue
        grouped[str(row["track_id"])].append(row)
    memories: list[TrackletMemory] = []
    for tracklet_id, items in grouped.items():
        frames = [int(item.get("frame", 0)) for item in items]
        team_counts = Counter(
            str(item.get("attributes", {}).get("team"))
            for item in items
            if item.get("attributes", {}).get("team") not in {None, "", -1, "-1"}
        )
        jersey_counts = Counter()
        role_counts = Counter(
            str(item.get("attributes", {}).get("role", "unknown")).lower() for item in items
        )
        for item in items:
            raw = item.get("attributes", {}).get("jersey")
            try:
                number = int(raw)
                if 0 <= number <= 99:
                    jersey_counts[number] += 1
            except (TypeError, ValueError):
                continue
        memories.append(
            TrackletMemory(
                tracklet_id=tracklet_id,
                match_id=match_id,
                team_posterior=_normalise_counter(team_counts),
                jersey_posterior=_normalise_counter(jersey_counts),
                role_posterior=_normalise_counter(role_counts),
                first_seen_ms=round((start_s + min(frames) / fps) * 1000),
                last_seen_ms=round((start_s + max(frames) / fps) * 1000),
            )
        )
    return memories


def _normalise_counter(counter: Counter) -> dict:
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def make_run_metadata(dataset: str, split: str, config: dict, models: dict) -> ModelRun:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()
    commit = "uncommitted"
    dirty = True
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
        dirty = True
    timestamp = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ_") + config_hash[:8]
    return ModelRun(
        run_id=run_id,
        timestamp=timestamp,
        git_commit=commit,
        dirty_worktree=dirty,
        dataset=dataset,
        split=split,
        config_hash=config_hash,
        models=models,
        prompt_versions={"semantic_event": "event-v1.0.0"},
        schema_versions={"event": "1.0.0", "player_memory": "1.0.0", "model_run": "1.0.0"},
        hardware={"platform": platform.platform(), "processor": platform.processor()},
        software={"python": platform.python_version(), "package": "football-intelligence==0.1.0"},
        seed=int(os.environ.get("FI_SEED", "0")),
    )
