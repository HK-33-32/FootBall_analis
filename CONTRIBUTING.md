# Contributing

Start with `README.md`, `ACCURACY.md`, `AGENTS.md` and `docs/CURRENT_STATE.md`.
Changes must preserve evidence provenance and must not turn heuristic metrics into
ground truth. Raw videos, datasets, checkpoints, secrets and generated runs are not
accepted in Git.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python scripts/check_public_repository.py
python -m pytest
docker compose config --quiet
docker build -t football-intelligence:local .
```

For accuracy or performance changes, include the dataset split, immutable model
revision, resolved configuration, command, hardware, output artifact location and
the matching entry in `docs/EXPERIMENTS.md`. Never tune on a frozen test split.

Keep pull requests focused. Explain the measurable failure being addressed and
the before/after result. A UI label must describe what the backend actually
measures and expose material uncertainty or coverage.

If a runtime dependency changes, update both `pyproject.toml` and the resolved
Linux/Python 3.12 `requirements-docker.lock`, rebuild the container, and verify
`pip check` inside it. Dependency-only updates are experiments when model output
or numerical behavior may change.
