"""Atomic content-addressed cache for expensive structured inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class InferenceCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.root / key[:2] / f"{key}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        directory = self.root / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return target
