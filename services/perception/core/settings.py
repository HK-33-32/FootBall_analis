"""Every path and knob the core reads from the environment.

The core is meant to be embedded: a service wraps it, a container runs it, and
neither should have to patch source to move a directory.  So everything that
differs between a laptop and a deployment lives here and comes from environment
variables, with defaults that work for a plain `git clone` on the host.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- layout -----------------------------------------------------------------
ENGINE_DIR = _path("FG_ENGINE_DIR", CORE_ROOT / "engine")
# Model weights are gigabytes and never change; they are mounted, not baked into
# the image, so the same image serves every deployment.
CHECKPOINT_DIR = _path("FG_CHECKPOINT_DIR", ENGINE_DIR / "checkpoints")
STORAGE_DIR = _path("FG_STORAGE_DIR", CORE_ROOT / "storage")
VIDEO_DIR = _path("FG_VIDEO_DIR", STORAGE_DIR / "videos")
JOB_DIR = _path("FG_JOB_DIR", STORAGE_DIR / "jobs")
ROSTER_DIR = _path("FG_ROSTER_DIR", CORE_ROOT / "rosters")
RUNNERS_DIR = CORE_ROOT / "core" / "runners"

PYTHON = Path(os.environ.get("FG_PYTHON") or sys.executable)

# --- server -----------------------------------------------------------------
HOST = os.environ.get("FG_HOST", "0.0.0.0")
PORT = _int("FG_PORT", 8000)
API_KEY = os.environ.get("FG_API_KEY", "")          # empty disables auth
MAX_UPLOAD_MB = _int("FG_MAX_UPLOAD_MB", 8192)
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("FG_CORS_ORIGINS", "*").split(",") if o.strip()]

# --- processing defaults ----------------------------------------------------
# Segment length caps peak disk and memory: frames of a finished segment are
# deleted before the next one starts, so a three-hour match costs the same as a
# one-minute clip.
SEGMENT_SECONDS = _float("FG_SEGMENT_SECONDS", 60.0)
VIS_WORKERS = _int("FG_VIS_WORKERS", 4)
CALIB_WORKERS_MAX = _int("FG_CALIB_WORKERS_MAX", 4)
RAM_PER_WORKER_GB = _float("FG_RAM_PER_WORKER_GB", 2.5)
RAM_RESERVE_GB = _float("FG_RAM_RESERVE_GB", 7.0)
QUEUE_WORKERS = _int("FG_QUEUE_WORKERS", 1)         # GPU jobs do not parallelise

DEFAULTS = {
    "fps": _int("FG_DEFAULT_FPS", 12),
    "resolution": os.environ.get("FG_DEFAULT_RESOLUTION", "1080"),
    "detector": os.environ.get("FG_DEFAULT_DETECTOR", "rfdetr"),
    "detector_size": os.environ.get("FG_DEFAULT_DETECTOR_SIZE", "large"),
    "jersey_reader": os.environ.get("FG_DEFAULT_JERSEY_READER", "qwen-vl"),
    "jersey_stride": _int("FG_DEFAULT_JERSEY_STRIDE", 3),
    "calib_stride": _int("FG_DEFAULT_CALIB_STRIDE", 1),
    "render_video": _flag("FG_DEFAULT_RENDER_VIDEO", False),
    "render_style": os.environ.get("FG_DEFAULT_RENDER_STYLE", "fifa"),
    "analytics": _flag("FG_DEFAULT_ANALYTICS", True),
}

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts"}


def ensure_dirs() -> None:
    for directory in (STORAGE_DIR, VIDEO_DIR, JOB_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> dict:
    """What the process actually resolved to — the first thing to check when a
    container behaves differently from a host run."""
    return {
        "engine_dir": str(ENGINE_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "storage_dir": str(STORAGE_DIR),
        "video_dir": str(VIDEO_DIR),
        "job_dir": str(JOB_DIR),
        "roster_dir": str(ROSTER_DIR),
        "python": str(PYTHON),
        "auth_enabled": bool(API_KEY),
        "segment_seconds": SEGMENT_SECONDS,
        "defaults": DEFAULTS,
    }
