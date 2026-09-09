"""HTTP surface of the analysis core.

The core is headless on purpose: it ingests a video, runs perception, derives an
event ledger and a per-player report, and hands them back as files and JSON.
Whatever renders that for a human — a web app, a dashboard, a Telegram bot —
lives outside and talks to this.

Everything long-running is a job: POST returns immediately with an id, the
caller either polls or supplies a callback URL.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import settings
from .jobs import STAGES, manager, stages_for
from .paths import ffprobe, human_size
from .schemas import (AnalysisRequest, HealthReport, JobDetail, JobSummary,
                      RosterInfo, VideoInfo)

VERSION = "1.0.0"
CHUNK = 1024 * 1024

app = FastAPI(
    title="Football Analysis Core",
    version=VERSION,
    description=(
        "Turns match video into an Event Ledger and per-player statistics.\n\n"
        "Every statistic traces back to an event, and every event carries the "
        "frame it happened on, so any number in the report can be replayed and "
        "checked. Counting statistics come from the ledger; distance and speed "
        "come from tracking and say so in their `source` field."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# --- auth -------------------------------------------------------------------
def require_key(x_api_key: Optional[str] = Header(None)) -> None:
    """No key configured means an open deployment; that is a deliberate choice
    for a core that normally sits behind a service on a private network."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(401, "неверный или отсутствующий X-API-Key")


guard = [Depends(require_key)]


# --- helpers ----------------------------------------------------------------
def _video_meta_path(video_id: str) -> Path:
    return settings.VIDEO_DIR / ("%s.json" % video_id)


def _load_video(video_id: str) -> dict:
    path = _video_meta_path(video_id)
    if not path.exists():
        raise HTTPException(404, "видео %s не найдено" % video_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _job_or_404(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "задача %s не найдена" % job_id)
    return job


def _artifact(job_id: str, name: str, media: str, filename: str) -> FileResponse:
    path = settings.JOB_DIR / job_id / name
    if not path.exists():
        raise HTTPException(404, "%s ещё не готов" % name)
    return FileResponse(str(path), media_type=media, filename=filename)


def _stream(path: Path, request: Request):
    """Range-aware streaming so a browser can seek in the annotated video."""
    if not path.exists():
        raise HTTPException(404, "файл не найден")
    size = path.stat().st_size
    media = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    header = request.headers.get("range")
    if not header:
        return FileResponse(str(path), media_type=media)
    match = re.match(r"bytes=(\d+)-(\d*)", header)
    if not match:
        raise HTTPException(416, "некорректный Range")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else size - 1
    end = min(end, size - 1)

    def body():
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                data = fh.read(min(CHUNK, left))
                if not data:
                    break
                left -= len(data)
                yield data

    return StreamingResponse(body(), status_code=206, media_type=media, headers={
        "Content-Range": "bytes %d-%d/%d" % (start, end, size),
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    })


def _notify(url: str, payload: dict) -> None:
    """Fire-and-forget callback; a deaf listener must not fail the job."""
    try:
        import urllib.request
        request = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(request, timeout=15).close()
    except Exception:
        pass


# --- videos -----------------------------------------------------------------
@app.post("/v1/videos", response_model=VideoInfo, dependencies=guard, tags=["видео"])
async def upload_video(file: UploadFile = File(...)) -> VideoInfo:
    """Upload a video and get the id every analysis request refers to."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in settings.VIDEO_SUFFIXES:
        raise HTTPException(400, "неподдерживаемый формат %s" % suffix)
    settings.ensure_dirs()
    video_id = "vid" + uuid.uuid4().hex[:12]
    target = settings.VIDEO_DIR / ("%s%s" % (video_id, suffix))
    written = 0
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    with open(target, "wb") as out:
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "файл больше %d МБ" % settings.MAX_UPLOAD_MB)
            out.write(chunk)

    probe = ffprobe(target)
    meta = {"id": video_id, "name": file.filename or target.name,
            "path": str(target), "size": written, **probe}
    _video_meta_path(video_id).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return VideoInfo(**{k: meta.get(k, 0) for k in VideoInfo.model_fields})


@app.post("/v1/videos/register", response_model=VideoInfo, dependencies=guard,
          tags=["видео"])
def register_video(path: str = Form(...), name: str = Form("")) -> VideoInfo:
    """Register a file the core can already reach, instead of uploading it.

    For a service that keeps media on a shared volume this avoids copying tens
    of gigabytes through HTTP just to hand over a path.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise HTTPException(404, "файл %s недоступен процессу ядра" % path)
    if source.suffix.lower() not in settings.VIDEO_SUFFIXES:
        raise HTTPException(400, "неподдерживаемый формат %s" % source.suffix)
    settings.ensure_dirs()
    video_id = "vid" + uuid.uuid4().hex[:12]
    probe = ffprobe(source)
    meta = {"id": video_id, "name": name or source.name, "path": str(source),
            "size": source.stat().st_size, **probe}
    _video_meta_path(video_id).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return VideoInfo(**{k: meta.get(k, 0) for k in VideoInfo.model_fields})


@app.get("/v1/videos", response_model=list[VideoInfo], dependencies=guard,
         tags=["видео"])
def list_videos() -> list[VideoInfo]:
    out = []
    for meta_path in sorted(settings.VIDEO_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append(VideoInfo(**{k: meta.get(k, 0) for k in VideoInfo.model_fields}))
    return out


@app.delete("/v1/videos/{video_id}", dependencies=guard, tags=["видео"])
def delete_video(video_id: str) -> dict:
    meta = _load_video(video_id)
    path = Path(meta["path"])
    # only files the core itself stored are removed; a registered path belongs
    # to whoever registered it
    if path.parent == settings.VIDEO_DIR:
        path.unlink(missing_ok=True)
    _video_meta_path(video_id).unlink(missing_ok=True)
    return {"deleted": video_id}


# --- jobs -------------------------------------------------------------------
@app.post("/v1/jobs", response_model=JobDetail, dependencies=guard, tags=["анализ"])
def create_job(request: AnalysisRequest) -> JobDetail:
    """Queue an analysis. Returns at once; poll or supply `callback_url`."""
    meta = _load_video(request.video_id)
    duration = float(meta.get("duration") or 0)
    end = min(request.end, duration) if duration else request.end
    if end - request.start < 0.2:
        raise HTTPException(400, "интервал вне длительности видео")

    roster = request.roster
    if roster is None and request.roster_id:
        path = settings.ROSTER_DIR / ("%s.json" % request.roster_id)
        if not path.exists():
            raise HTTPException(404, "заявка %s не найдена" % request.roster_id)
        roster = json.loads(path.read_text(encoding="utf-8"))

    defaults = settings.DEFAULTS
    job = manager.create(
        video=meta, start=request.start, end=end,
        fps=request.fps or defaults["fps"],
        jersey_mode="CLIP",
        resolution=request.resolution or defaults["resolution"],
        detector=request.detector or defaults["detector"],
        detector_size=request.detector_size or defaults["detector_size"],
        jersey_stride=request.jersey_stride or defaults["jersey_stride"],
        calib_stride=request.calib_stride or defaults["calib_stride"],
        jersey_reader=request.jersey_reader or defaults["jersey_reader"],
        segment_seconds=request.segment_seconds or settings.SEGMENT_SECONDS,
        roster=json.dumps(roster, ensure_ascii=False) if roster else "",
        render_video=(defaults["render_video"] if request.render_video is None
                      else request.render_video),
        render_style=request.render_style or defaults["render_style"],
        analytics=(defaults["analytics"] if request.analytics is None
                   else request.analytics),
    )
    job.metadata = dict(request.metadata)
    job.kickoff_offset_s = request.kickoff_offset_s
    job.period = request.period
    if request.callback_url:
        _watch_for_callback(job.id, request.callback_url)
    return _detail(job)


def _watch_for_callback(job_id: str, url: str) -> None:
    """Poll our own job table in a daemon thread and POST once it settles."""
    def wait():
        while True:
            job = manager.get(job_id)
            if job is None:
                return
            if job.status not in ("queued", "running"):
                _notify(url, _detail(job).model_dump())
                return
            time.sleep(2.0)

    threading.Thread(target=wait, daemon=True, name="callback-%s" % job_id).start()


def _detail(job) -> JobDetail:
    data = job.as_dict()
    data["log"] = job.tail(200)
    data["metadata"] = getattr(job, "metadata", {}) or {}
    data["stages"] = [{"key": s.key, "title": s.title, "weight": s.weight}
                      for s in stages_for(job)]
    allowed = set(JobDetail.model_fields)
    return JobDetail(**{k: v for k, v in data.items() if k in allowed})


@app.get("/v1/jobs", response_model=list[JobSummary], dependencies=guard,
         tags=["анализ"])
def list_jobs(status: Optional[str] = None, limit: int = 50) -> list[JobSummary]:
    allowed = set(JobSummary.model_fields)
    out = []
    for data in manager.list():
        if status and data.get("status") != status:
            continue
        out.append(JobSummary(**{k: v for k, v in data.items() if k in allowed}))
    return out[:limit]


@app.get("/v1/jobs/{job_id}", response_model=JobDetail, dependencies=guard,
         tags=["анализ"])
def job_detail(job_id: str) -> JobDetail:
    return _detail(_job_or_404(job_id))


@app.post("/v1/jobs/{job_id}/cancel", dependencies=guard, tags=["анализ"])
def cancel_job(job_id: str) -> dict:
    _job_or_404(job_id).cancel()
    return {"canceled": job_id}


@app.delete("/v1/jobs/{job_id}", dependencies=guard, tags=["анализ"])
def delete_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job and job.status == "running":
        raise HTTPException(409, "задача выполняется; сначала /cancel")
    shutil.rmtree(settings.JOB_DIR / job_id, ignore_errors=True)
    manager.jobs.pop(job_id, None)
    return {"deleted": job_id}


# --- results ----------------------------------------------------------------
@app.get("/v1/jobs/{job_id}/events", dependencies=guard, tags=["результаты"])
def job_events(job_id: str):
    """The Event Ledger: every statistic in the report expands into these."""
    return _artifact(job_id, "events.json", "application/json",
                     "events_%s.json" % job_id)


@app.get("/v1/jobs/{job_id}/events.sqlite", dependencies=guard, tags=["результаты"])
def job_events_db(job_id: str):
    """The same ledger as an indexed SQLite database, for ad-hoc queries."""
    return _artifact(job_id, "events.sqlite", "application/vnd.sqlite3",
                     "events_%s.sqlite" % job_id)


@app.get("/v1/jobs/{job_id}/players", dependencies=guard, tags=["результаты"])
def job_players(job_id: str):
    """Per-player report: coverage, on-ball, passing, defensive, physical,
    spatial, and a timeline of every event with its timestamp."""
    return _artifact(job_id, "player_stats.json", "application/json",
                     "players_%s.json" % job_id)


@app.get("/v1/jobs/{job_id}/players.csv", dependencies=guard, tags=["результаты"])
def job_players_csv(job_id: str):
    return _artifact(job_id, "player_stats.csv", "text/csv",
                     "players_%s.csv" % job_id)


@app.get("/v1/jobs/{job_id}/predictions", dependencies=guard, tags=["результаты"])
def job_predictions(job_id: str):
    """Raw perception output: one row per detection, with pitch coordinates."""
    return _artifact(job_id, "predictions.json", "application/json",
                     "predictions_%s.json" % job_id)


@app.get("/v1/jobs/{job_id}/video", dependencies=guard, tags=["результаты"])
def job_video(job_id: str, request: Request):
    """The annotated video, when the job was asked to render one."""
    return _stream(settings.JOB_DIR / job_id / "result.mp4", request)


@app.post("/v1/jobs/{job_id}/reanalyse", dependencies=guard, tags=["результаты"])
def reanalyse(job_id: str, kickoff_offset_s: Optional[float] = Form(None),
              period: Optional[int] = Form(None)) -> dict:
    """Rebuild the ledger from a finished run without touching perception.

    The analytics layer reads `predictions.json` and loads no model, so tuning
    the derivation costs seconds instead of the hours perception took.
    """
    job_dir = settings.JOB_DIR / job_id
    if not (job_dir / "predictions.json").exists():
        raise HTTPException(404, "нет предсказаний для анализа")
    from .analytics.run import analyse
    report = analyse(str(job_dir), kickoff_offset_s=kickoff_offset_s,
                     period=period, verbose=False)
    return {"players": len(report["players"]), "teams": report["teams"],
            "identification": report["identification"]}


# --- rosters ----------------------------------------------------------------
@app.get("/v1/rosters", response_model=list[RosterInfo], dependencies=guard,
         tags=["справочники"])
def list_rosters() -> list[RosterInfo]:
    out = []
    if settings.ROSTER_DIR.is_dir():
        for path in sorted(settings.ROSTER_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(RosterInfo(
                id=path.stem, name=data.get("name", ""),
                teams=[t.get("team", "") for t in data.get("teams", [])]))
    return out


# --- health -----------------------------------------------------------------
@app.get("/v1/health", response_model=HealthReport, tags=["служебное"])
def health() -> HealthReport:
    checkpoints = {}
    for name in ("SoccernetGSR_EfficientNet_Best.pth", "CLIP_Jersey.pth",
                 "sports_model.pth.tar-60"):
        checkpoints[name] = (settings.CHECKPOINT_DIR / name).is_file()
    gpu, cuda = None, False
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        pass
    queue = {"queued": 0, "running": 0, "done": 0, "error": 0, "canceled": 0}
    for data in manager.list():
        queue[data.get("status", "done")] = queue.get(data.get("status"), 0) + 1
    ok = cuda and all(checkpoints.values()) and bool(shutil.which("ffmpeg"))
    return HealthReport(
        status="ok" if ok else "degraded", version=VERSION, gpu=gpu,
        cuda_available=cuda, ffmpeg=bool(shutil.which("ffmpeg")),
        checkpoints=checkpoints, settings=settings.describe(), queue=queue)


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({
        "name": "Football Analysis Core", "version": VERSION,
        "docs": "/docs", "openapi": "/openapi.json", "health": "/v1/health",
    })
