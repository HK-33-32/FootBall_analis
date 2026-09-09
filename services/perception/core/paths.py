"""Filesystem helpers, resolved from `settings` so the container can move them."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .settings import (CHECKPOINT_DIR, ENGINE_DIR, JOB_DIR, PYTHON, ROSTER_DIR,
                       RUNNERS_DIR, STORAGE_DIR, VIDEO_DIR, VIDEO_SUFFIXES,
                       ensure_dirs)

# names the orchestration already uses, pointed at the core layout
ROOT = ENGINE_DIR.parent
REPO = ENGINE_DIR
RUNNERS = RUNNERS_DIR
VENV_PYTHON = PYTHON

ensure_dirs()


def ffprobe(path: Path) -> dict:
    """Return duration / fps / size of a video file (empty dict on failure)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration,size",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    fps = 0.0
    raw = stream.get("avg_frame_rate") or "0/0"
    try:
        num, _, den = raw.partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
    except Exception:
        fps = 0.0
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": round(fps, 3),
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or (path.stat().st_size if path.exists() else 0)),
    }


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TB"

def free_ram_gb() -> float:
    """Physical memory currently available, in GB (0.0 if it cannot be read)."""
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 0.0
