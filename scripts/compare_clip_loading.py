"""Fail closed on CLIP state, dtype, preprocessing, output or checkpoint drift."""

import argparse
import json
from pathlib import Path


def compare(before, after):
    for run in (before, after):
        if run["status"] != "complete":
            raise ValueError("Both CLIP runs must be complete")
        if run["checkpoint"]["missing"] or run["checkpoint"]["unexpected"]:
            raise ValueError("Checkpoint is not a complete model state")
    for key in ("source_manifest_sha256", "script_sha256", "config_hash", "hardware", "software"):
        if before[key] != after[key]:
            raise ValueError(f"Incomparable CLIP experiment: {key}")
    if before["checkpoint"]["sha256"] != after["checkpoint"]["sha256"]:
        raise ValueError("Finetuned checkpoint differs")
    if not before["state"] or not before["inputs"] or not before["outputs"]:
        raise ValueError("Empty CLIP evidence")
    checks = {key: before[key] == after[key] for key in (
        "state", "inputs", "outputs", "requires_grad", "training_flags")}
    if min(before["load_s"], after["load_s"]) <= 0:
        raise ValueError("Load time must be positive")
    return {"baseline": before["run_id"], "candidate": after["run_id"],
            "exact": checks, "all_exact": all(checks.values()),
            "load_s": {"baseline": before["load_s"], "candidate": after["load_s"]},
            "load_speedup": before["load_s"] / after["load_s"],
            "note": "CLIP load only; equality includes all state values/dtypes, real crop "
                    "preprocessing, logits, color/text embeddings and train/freeze flags. "
                    "This is not a new accuracy score or whole-match timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    if not result["all_exact"]:
        raise SystemExit("CLIP acceptance failed; inspect saved differing evidence")


if __name__ == "__main__":
    main()
