"""Verify paired benchmark manifests and emit a machine-readable comparison."""

import argparse
import json
from pathlib import Path


def summarize(directory: Path) -> dict:
    comparisons = []
    for before_path in sorted(directory.glob("*_baseline_v2.json")):
        after_path = directory / before_path.name.replace("_baseline_v2", "_current_v2")
        if not after_path.is_file():
            raise ValueError(f"missing paired run: {after_path}")
        before = json.loads(before_path.read_text("utf-8"))
        after = json.loads(after_path.read_text("utf-8"))
        hashes = set(before["output_hashes"] + after["output_hashes"])
        same_inputs = all(
            before.get(key) == after.get(key)
            for key in (
                "input_sha256",
                "video_sha256",
                "config_hash",
                "frame_interval",
                "split",
                "dataset",
            )
        )
        comparisons.append(
            {
                "name": before_path.stem.removesuffix("_baseline_v2"),
                "baseline": str(before_path),
                "current": str(after_path),
                "dataset": before["dataset"],
                "split": before["split"],
                "baseline_decoder": before.get("decoder", "native"),
                "current_decoder": after.get("decoder", "native"),
                "before_median_s": before["median_s"],
                "after_median_s": after["median_s"],
                "speedup": before["median_s"] / after["median_s"],
                "same_inputs_and_metric_config": same_inputs,
                "exact_output_equivalence": len(hashes) == 1,
                "output_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            }
        )
    if not comparisons:
        raise ValueError("no paired benchmark runs")
    return {
        "schema_version": "1.0.0",
        "scope": "CPU postprocessing, not end-to-end GPU inference",
        "comparisons": comparisons,
        "all_equivalent": all(
            c["exact_output_equivalence"] and c["same_inputs_and_metric_config"]
            for c in comparisons
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(result['comparisons'])} comparisons; all_equivalent={result['all_equivalent']}")
    if not result["all_equivalent"]:
        raise SystemExit(1)
