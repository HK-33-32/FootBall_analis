"""Official GS-HOTA milestone with prediction, annotation and evaluator fingerprints."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from benchmark_gamestate import evaluate


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    manifest = args.runtime_manifest or args.current.parent / "manifest.json"
    for path in (args.current, args.historical, manifest):
        if not path.is_file():
            raise SystemExit(f"Required artifact is not ready: {path}")
    runtime = json.loads(manifest.read_text(encoding="utf-8"))
    if runtime.get("status") not in {"done", "complete"}:
        raise SystemExit("Current runtime/replay manifest is not complete")
    root, evaluator = Path("data/soccernet"), Path("data/tools/sn-trackeval")
    commit = subprocess.check_output(
        ["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(evaluator), "status", "--porcelain"], text=True,
    ).strip()
    if dirty:
        raise SystemExit("Official evaluator worktree is dirty; inspect before evaluation")
    results = {}
    for name, predictions in (("historical", args.historical), ("current", args.current)):
        tracker = f"{args.output.stem}-{name}"
        metrics = evaluate(predictions, args.sequence, tracker, root, evaluator)
        results[name] = {"predictions": str(predictions), "sha256": sha256(predictions),
                         "metrics": metrics}
    result = {
        "schema_version": "core-milestone.v1", "run_id": args.output.stem,
        "timestamp": datetime.now(UTC).isoformat(), "sequence": args.sequence,
        "split": "validation", "evaluator_commit": commit,
        "annotations_sha256": sha256(root / "valid" / args.sequence / "Labels-GameState.json"),
        "config": {"roles": True, "teams": True, "jerseys": True, "ignore_ball": True},
        "current_runtime_manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "results": results,
        "delta_gs_hota": results["current"]["metrics"]["HOTA"]
                         - results["historical"]["metrics"]["HOTA"],
        "limitations": ["Historical artifact comparison, not a newly controlled native A/B",
                        "Historical calibration may be uncontrolled; no non-inferiority claim",
                        "One validation sequence, no SOTA or aggregate claim"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: value["metrics"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
