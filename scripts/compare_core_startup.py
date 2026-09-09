"""Compare completed startup experiments without conflating them with whole jobs."""

import argparse
import json
import statistics
from pathlib import Path


def compare(before, after, allowed_removed_weights=()):
    for item in (before, after):
        if item["status"] != "complete":
            raise ValueError("Both startup runs must be complete")
    for key in ("video_sha256", "frame_hashes", "source_config_sha256", "split",
                "profile_init", "hardware", "software", "script_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"Incomparable startup runs: {key}")
    left_weights, right_weights = before["loaded_weight_hashes"], after["loaded_weight_hashes"]
    removed = left_weights.keys() - right_weights.keys()
    if (not removed <= set(allowed_removed_weights) or right_weights.keys() - left_weights.keys()
            or any(left_weights[key] != right_weights[key] for key in right_weights)):
        raise ValueError("Incomparable startup runs: loaded_weight_hashes")
    if not before["outputs"] or not after["outputs"]:
        raise ValueError("Tracking output is missing")
    timings = {}
    for stage in ("imports", "initialization", "tracking"):
        left, right = before["stages"][stage], after["stages"][stage]
        if left <= 0 or right <= 0:
            raise ValueError("Stage time must be positive")
        timings[stage] = {"before_s": left, "after_s": right, "speedup": left / right}
    total_before = sum(row["before_s"] for row in timings.values())
    total_after = sum(row["after_s"] for row in timings.values())
    return {
        "baseline": before["run_id"], "candidate": after["run_id"],
        "exact_tracking_output": before["outputs"] == after["outputs"],
        "explicitly_removed_weight_files": sorted(removed),
        "changed_sources": sorted(key for key in before["source_hashes"] | after["source_hashes"]
                                  if before["source_hashes"].get(key)
                                  != after["source_hashes"].get(key)),
        "timings": timings,
        "timed_tracking_stages": {"before_s": total_before, "after_s": total_after,
                                  "speedup": total_before / total_after},
        "note": "Sum of imports + initialization + tracking; excludes extraction, "
                "benchmark fingerprinting, concurrent calibration, IDATR and full-job overhead.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-removed-weight", action="append", default=[],
                        help="Explicit redundant file removal; all remaining hashes must match")
    args = parser.parse_args()
    if len(args.baseline) != len(args.candidate):
        raise SystemExit("Provide the same number of baseline and candidate manifests")
    before = [json.loads(path.read_text()) for path in args.baseline]
    after = [json.loads(path.read_text()) for path in args.candidate]
    pairs = [compare(left, right, args.allow_removed_weight)
             for left, right in zip(before, after, strict=True)]
    result = pairs[0] if len(pairs) == 1 else aggregate(before, after, args.allow_removed_weight)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


def aggregate(before, after, allowed_removed_weights=()):
    if not before or len(before) != len(after):
        raise ValueError("Require equal nonempty paired runs")
    pairs = [compare(left, right, allowed_removed_weights)
             for left, right in zip(before, after, strict=True)]
    cross = [compare(before[0], item, allowed_removed_weights) for item in [*before[1:], *after]]
    timings = {}
    for stage in ("imports", "initialization", "tracking", "timed_tracking_stages"):
        rows = [pair[stage] if stage == "timed_tracking_stages" else pair["timings"][stage]
                for pair in pairs]
        left = [row["before_s"] for row in rows]
        right = [row["after_s"] for row in rows]
        timings[stage] = {"baseline_samples_s": left, "candidate_samples_s": right,
                          "baseline_median_s": statistics.median(left),
                          "candidate_median_s": statistics.median(right),
                          "median_speedup": statistics.median(left) / statistics.median(right)}
    return {"pairs": pairs, "all_tracking_outputs_identical": all(
        row["exact_tracking_output"] for row in cross), "timings": timings,
        "sample_pairs": len(pairs), "note": pairs[0]["note"]}


if __name__ == "__main__":
    main()
