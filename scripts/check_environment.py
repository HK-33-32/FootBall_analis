from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import httpx


def check_url(url: str) -> str:
    if not url:
        return "not configured"
    try:
        response = httpx.get(url, timeout=5)
        return f"HTTP {response.status_code}"
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    legacy = Path(os.environ.get("FI_LEGACY_WEIGHTS", "../../Football_grade/football_core/weights"))
    checkpoints = legacy / "checkpoints"
    expected = [
        "SoccernetGSR_EfficientNet_Best.pth",
        "CLIP_Jersey.pth",
        "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    ]
    print(
        json.dumps(
            {
                "ffmpeg": shutil.which("ffmpeg"),
                "legacy_weights": str(legacy.resolve()),
                "checkpoints": {name: (checkpoints / name).is_file() for name in expected},
                "perception": check_url(
                    os.environ.get("FI_PERCEPTION_URL", "").rstrip("/") + "/v1/health"
                    if os.environ.get("FI_PERCEPTION_URL")
                    else ""
                ),
                "vlm": check_url(
                    os.environ.get("FI_VLM_BASE_URL", "").rstrip("/") + "/v1/models"
                    if os.environ.get("FI_VLM_BASE_URL")
                    else ""
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
