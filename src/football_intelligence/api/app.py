"""FastAPI control plane and debugging frontend."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..analysis import AnalysisConfig, AnalysisManager, CoreClient
from ..cache import InferenceCache
from ..config import load_config
from ..events import ActiveSemanticEngine, ReanalysisConfig
from ..identity import GlobalIdentitySolver, IdentityConfig
from ..memory import MatchMemory
from ..perception import LegacyCoreClient
from ..pipeline import MatchPipeline
from ..rosters import MatchRoster, RosterStore, install_bundled_rosters
from ..viewer import library
from ..vlm import OpenAICompatibleVLM


class JobRequest(BaseModel):
    match_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    video_path: str
    start_s: float = Field(default=0, ge=0)
    end_s: float = Field(gt=0)
    fps: int = Field(default=12, ge=1, le=30)
    legacy_events_path: str | None = None
    predictions_path: str | None = None
    roster: MatchRoster | None = None
    use_starting_lineup: bool = False
    render_video: bool = True
    run_semantics: bool = True
    dataset: str = "user-media"
    split: str = "unlabelled"

    @model_validator(mode="after")
    def interval(self) -> JobRequest:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if bool(self.legacy_events_path) != bool(self.predictions_path):
            raise ValueError("legacy_events_path and predictions_path must be supplied together")
        return self


class JobManager:
    def __init__(self, pipeline: MatchPipeline, root: Path):
        self.pipeline = pipeline
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="football-pipeline")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self._restore()

    def _restore(self) -> None:
        for state_path in self.root.glob("*/state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["status"] == "running":
                    state.update(
                        status="interrupted",
                        error="process stopped; resubmit to resume from cached perception",
                    )
                self.jobs[state["job_id"]] = state
            except (OSError, json.JSONDecodeError, KeyError):
                continue

    def submit(self, request: JobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        state = {
            "job_id": job_id,
            "match_id": request.match_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "error": None,
            "created_at": datetime.now(UTC).isoformat(),
            "request": request.model_dump(),
        }
        with self.lock:
            self.jobs[job_id] = state
            self._save(state)
        self.executor.submit(self._run, job_id, request)
        return dict(state)

    def _save(self, state: dict[str, Any]) -> None:
        directory = self.root / state["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "state.json.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "state.json")

    def _update(self, job_id: str, **changes: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(changes)
            self._save(self.jobs[job_id])

    def _run(self, job_id: str, request: JobRequest) -> None:
        self._update(job_id, status="running", stage="starting", progress=0.01)
        try:
            legacy_events = (
                _read_json(request.legacy_events_path) if request.legacy_events_path else None
            )
            predictions = _read_json(request.predictions_path) if request.predictions_path else None
            result = self.pipeline.run(
                match_id=request.match_id,
                video_path=request.video_path,
                start_s=request.start_s,
                end_s=request.end_s,
                fps=request.fps,
                legacy_events=legacy_events,
                predictions=predictions,
                roster=(
                    request.roster.perception_payload(request.use_starting_lineup)
                    if request.roster
                    else None
                ),
                render_video=request.render_video,
                run_semantics=request.run_semantics,
                dataset=request.dataset,
                split=request.split,
                progress=lambda stage, value: self._update(job_id, stage=stage, progress=value),
            )
            if (
                request.render_video
                and result.get("legacy_job")
                and self.pipeline.perception is not None
            ):
                annotated = self.root / job_id / "annotated.mp4"
                self.pipeline.perception.download_artifact(
                    result["legacy_job"]["id"], "video", annotated
                )
                result["annotated_video"] = f"/api/v1/jobs/{job_id}/annotated-video"
            result_path = self.root / job_id / "result.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._update(
                job_id, status="done", stage="done", progress=1.0, result_path=str(result_path)
            )
        except Exception as exc:
            self._update(
                job_id, status="error", stage="error", error=f"{type(exc).__name__}: {exc}"
            )

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(
                (dict(state) for state in self.jobs.values()),
                key=lambda row: row["created_at"],
                reverse=True,
            )


class AnalysisRequest(BaseModel):
    """What starting a calculation needs. Declared here rather than inside the
    app factory: with postponed annotations FastAPI resolves the type by name
    against the module, and a class defined in a function is not there -- it
    silently becomes a query parameter instead of the body."""

    video: str
    title: str = ""
    roster: dict[str, Any] | None = None


def _read_json(path: str) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    root = Path(data_dir or os.environ.get("FI_DATA_DIR", "./data/runtime")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    memory = MatchMemory(root / "match_memory.sqlite")
    config_path = Path(os.environ.get("FI_CONFIG", "configs/default.yaml"))
    if not config_path.is_absolute():
        repository_config = Path(__file__).resolve().parents[3] / config_path
        config_path = repository_config if repository_config.is_file() else config_path
    config = load_config(config_path) if config_path.is_file() else {}
    perception_url = os.environ.get("FI_PERCEPTION_URL", "").strip()
    perception = (
        LegacyCoreClient(perception_url, os.environ.get("FI_PERCEPTION_API_KEY", ""))
        if perception_url
        else None
    )
    vlm_url = os.environ.get("FI_VLM_BASE_URL", "").strip()
    semantic_engine = None
    if vlm_url:
        first = OpenAICompatibleVLM(
            base_url=vlm_url,
            model_name=os.environ.get("FI_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            model_revision=os.environ.get(
                "FI_VLM_REVISION", "a115a837cf3cbba4b697aa74b609721b5009ed41"
            ),
            api_key=os.environ.get("FI_VLM_API_KEY", "EMPTY"),
        )
        semantic_config = config.get("semantic", {})
        semantic_engine = ActiveSemanticEngine(
            first,
            config=ReanalysisConfig(
                event_confidence_threshold=semantic_config.get("event_confidence_threshold", 0.68),
                alternative_margin_threshold=semantic_config.get(
                    "alternative_margin_threshold", 0.18
                ),
                first_pass_fps=semantic_config.get("first_pass_fps", 1.5),
                second_pass_fps=semantic_config.get("second_pass_fps", 4.0),
                max_frames_first=semantic_config.get("max_frames_first", 10),
                max_frames_second=semantic_config.get("max_frames_second", 24),
            ),
            cache=InferenceCache(root / "cache" / "vlm"),
        )
    identity_config = config.get("identity", {})
    pipeline = MatchPipeline(
        memory=memory,
        semantic_engine=semantic_engine,
        perception=perception,
        identity_solver=GlobalIdentitySolver(
            IdentityConfig(
                reid_weight=identity_config.get("reid_weight", 0.55),
                team_weight=identity_config.get("team_weight", 0.20),
                jersey_weight=identity_config.get("jersey_weight", 0.25),
                merge_threshold=identity_config.get("merge_threshold", 0.70),
                overlap_tolerance_ms=identity_config.get("overlap_tolerance_ms", 200),
                suppress_imbalanced_team_evidence=identity_config.get(
                    "suppress_imbalanced_team_evidence", True
                ),
                min_team_share=identity_config.get("min_team_share", 0.20),
                min_team_tracklets=identity_config.get("min_team_tracklets", 8),
            )
        ),
    )
    manager = JobManager(pipeline, root / "jobs")
    roster_store = RosterStore(root / "rosters")
    install_bundled_rosters(roster_store)

    application = FastAPI(title="Football Match Intelligence", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in os.environ.get("FI_CORS_ORIGINS", "*").split(",")
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.memory = memory
    application.state.manager = manager
    application.state.root = root
    application.state.perception = perception
    application.state.semantic_engine = semantic_engine
    application.state.roster_store = roster_store

    @application.get("/api/v1/health")
    def health() -> dict[str, Any]:
        perception_health: dict[str, Any] | str = "not_configured"
        if perception:
            try:
                perception_health = perception.health()
            except Exception as exc:
                perception_health = f"unavailable: {type(exc).__name__}: {exc}"
        return {
            "status": "ok",
            "version": "0.1.0",
            "perception": perception_health,
            "semantic_vlm": {
                "configured": semantic_engine is not None,
                "model": semantic_engine.first_pass.model_name if semantic_engine else None,
                "revision": semantic_engine.first_pass.model_revision if semantic_engine else None,
            },
            "database": str(memory.path),
            "config": str(config_path),
        }

    @application.post("/api/v1/uploads")
    async def upload(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}:
            raise HTTPException(415, "unsupported video extension")
        target = root / "uploads" / f"{uuid.uuid4().hex}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output, length=1024 * 1024)
        return {"path": str(target), "name": file.filename, "size": target.stat().st_size}

    @application.get("/api/v1/rosters")
    def list_rosters() -> list[dict]:
        return roster_store.list()

    @application.post("/api/v1/rosters", status_code=201)
    def save_roster(roster: MatchRoster) -> dict:
        return roster_store.save(roster)

    @application.get("/api/v1/rosters/{roster_id}")
    def get_roster(roster_id: str) -> dict:
        try:
            return roster_store.load(roster_id).model_dump()
        except KeyError:
            raise HTTPException(404, "roster not found") from None

    @application.post("/api/v1/jobs", status_code=202)
    def submit_job(request: JobRequest) -> dict[str, Any]:
        return manager.submit(request)

    @application.get("/api/v1/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return manager.list()

    @application.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found") from None

    @application.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict[str, Any]:
        try:
            state = manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found") from None
        if state["status"] != "done":
            raise HTTPException(409, f"job is {state['status']}")
        return _read_json(state["result_path"])

    @application.get("/api/v1/jobs/{job_id}/annotated-video")
    def annotated_video(job_id: str):
        try:
            manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found") from None
        source = root / "jobs" / job_id / "annotated.mp4"
        if not source.is_file():
            raise HTTPException(404, "annotated video was not rendered")
        return FileResponse(source, media_type="video/mp4")

    @application.get("/api/v1/matches/{match_id}/events")
    def events(match_id: str, event_type: str | None = None) -> list[dict]:
        return memory.events(match_id, event_type)

    @application.get("/api/v1/matches/{match_id}/players")
    def players(match_id: str) -> list[dict]:
        return memory.players(match_id)

    @application.get("/api/v1/matches/{match_id}/statistics")
    def statistics(match_id: str) -> dict:
        return memory.statistics(match_id)

    @application.get("/api/v1/evidence/{evidence_id}")
    def evidence(evidence_id: str) -> dict:
        result = memory.evidence(evidence_id)
        if not result:
            raise HTTPException(404, "evidence not found")
        return result

    @application.get("/api/v1/evidence/{evidence_id}/media")
    def evidence_media(evidence_id: str):
        result = memory.evidence(evidence_id)
        if not result:
            raise HTTPException(404, "evidence not found")
        source = Path(result["source_uri"]).resolve()
        if not source.is_file():
            raise HTTPException(404, "source media is no longer available")
        return FileResponse(source)

    # ------------------------------------------------------------- viewer
    # Finished matches are read from a mounted directory rather than baked into
    # a page: the clip streams, so seeking works and the page stays small.
    reports_root = Path(os.environ.get("FI_REPORTS_DIR", root / "reports"))
    application.state.reports_root = reports_root

    analysis_config = AnalysisConfig(
        core_url=os.environ.get("FI_CORE_URL", "http://host.docker.internal:8000"),
        media_dir=Path(os.environ.get("FI_MEDIA_DIR", "/data/perception_benchmark/media")),
        media_container_root=os.environ.get("FI_MEDIA_CONTAINER_ROOT", "/data/media"),
        reports_dir=reports_root,
        fps=int(os.environ.get("FI_ANALYSIS_FPS", "25")),
        chunk_s=float(os.environ.get("FI_CHUNK_SECONDS", "60")),
    )
    analyses = AnalysisManager(analysis_config)
    application.state.analyses = analyses

    def _require_report(report_id: str) -> library.Report:
        found = library.find(reports_root, report_id)
        if found is None:
            raise HTTPException(404, "no such match report")
        return found

    @application.get("/api/v1/analysis/backend")
    def analysis_backend() -> dict[str, Any]:
        """Whether a calculation can be started at all, and where it would run."""
        client = CoreClient(analysis_config.core_url)
        return {
            "core_url": analysis_config.core_url,
            "reachable": client.healthy(),
            "reports_writable": os.access(reports_root, os.W_OK),
            "media_dir": str(analysis_config.media_dir),
            "chunk_s": analysis_config.chunk_s,
        }

    @application.post("/api/v1/analyses", status_code=202)
    def start_analysis(request: AnalysisRequest) -> dict[str, Any]:
        video = Path(request.video)
        if not video.is_file():
            raise HTTPException(404, f"видеофайл не найден: {video}")
        if not os.access(reports_root, os.W_OK):
            raise HTTPException(
                409,
                f"каталог отчётов {reports_root} только для чтения — "
                "смонтируйте его на запись, чтобы запускать расчёт",
            )
        roster = None
        if request.roster:
            # validated the same way a saved roster is, so a typo is refused
            # here rather than halfway through an hour of GPU
            try:
                checked = MatchRoster.model_validate(request.roster)
            except ValidationError as exc:
                # pydantic's own error objects are not JSON: keep the part a
                # person needs, which is where it went wrong and why
                problems = [
                    {"field": ".".join(str(part) for part in item["loc"]), "problem": item["msg"]}
                    for item in exc.errors(include_url=False)
                ]
                raise HTTPException(422, problems) from exc
            roster_store.save(checked)
            roster = checked.model_dump()
        analysis = analyses.submit(video, request.title, roster)
        return analysis.as_dict()

    @application.get("/api/v1/analyses")
    def list_analyses() -> list[dict[str, Any]]:
        return analyses.list()

    @application.get("/api/v1/analyses/{analysis_id}")
    def get_analysis(analysis_id: str) -> dict[str, Any]:
        analysis = analyses.get(analysis_id)
        if analysis is None:
            raise HTTPException(404, "нет такого расчёта")
        return analysis.as_dict()

    @application.post("/api/v1/analyses/{analysis_id}/cancel")
    def cancel_analysis(analysis_id: str) -> dict[str, Any]:
        analysis = analyses.get(analysis_id)
        if analysis is None:
            raise HTTPException(404, "нет такого расчёта")
        analysis.cancel()
        return analysis.as_dict()

    @application.get("/api/v1/reports")
    def list_reports() -> list[dict[str, Any]]:
        return [item.as_dict() for item in library.discover(reports_root)]

    @application.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str) -> dict[str, Any]:
        return json.loads(_require_report(report_id).path.read_text(encoding="utf-8"))

    @application.get("/api/v1/reports/{report_id}/video")
    def report_video(report_id: str):
        found = _require_report(report_id)
        if found.video is None:
            raise HTTPException(404, "this match has no clip beside its report")
        return FileResponse(found.video)

    @application.get("/matches/{report_id}", response_class=HTMLResponse)
    def match_viewer(report_id: str) -> HTMLResponse:
        found = _require_report(report_id)
        report = json.loads(found.path.read_text(encoding="utf-8"))
        video = f"/api/v1/reports/{report_id}/video" if found.video else ""
        return HTMLResponse(library.render(report, video, title=found.title))

    viewer_dir = Path(library.TEMPLATE_PATH).parent
    static = Path(__file__).parent / "static"

    @application.get("/", response_class=HTMLResponse)
    def match_library() -> HTMLResponse:
        return HTMLResponse((viewer_dir / "index.html").read_text(encoding="utf-8"))

    @application.get("/lab", response_class=HTMLResponse)
    def evidence_lab() -> HTMLResponse:
        """The older debugging frontend, kept for the job-level pipeline."""
        return HTMLResponse((static / "index.html").read_text(encoding="utf-8"))

    application.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return application


app = create_app()
