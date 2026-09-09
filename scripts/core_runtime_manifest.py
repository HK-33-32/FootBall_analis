"""Run inside the perception container: identify code, available weights and runtime."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


roots = [Path("/opt/weights"), Path("/root/.cache/clip")]
weights = {}
for root in roots:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".gguf"}:
            weights[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
source = {}
for directory in ("/app/core", "/app/engine"):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            source[str(path)] = sha256(path)
packages = {}
for name in ("torch", "torchvision", "transformers", "numpy", "rfdetr", "opencv-python"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "models": weights,
    "model_inventory_note": "Available files; effective config selects used models",
    "source": source, "software": {"python": platform.python_version(), **packages},
    "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
    "runtime_environment": {name: os.environ.get(name) for name in (
        "FG_CALIB_SEED", "FG_CALIB_WORKERS_MAX", "FG_QUEUE_WORKERS",
    )},
}))
