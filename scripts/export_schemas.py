from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.domain import GlobalPlayerMemory, ModelRun, SemanticEvent


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(parents=True, exist_ok=True)
    for name, model in {
        "event.schema.json": SemanticEvent,
        "player_memory.schema.json": GlobalPlayerMemory,
        "model_run.schema.json": ModelRun,
    }.items():
        (root / name).write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
