"""Compare complete core runs, retaining identity, geometry and attribute differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def canonical_rows(document: dict) -> list[dict]:
    rows = []
    for original in document["predictions"]:
        if "frame" not in original:
            raise ValueError("Explicit source frame required for cross-job comparison")
        # These three IDs are generated from a new job/clip name, not identities.
        rows.append({key: value for key, value in original.items()
                     if key not in {"id", "image_id", "video_id"}})
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def compare_predictions(before: dict, after: dict) -> dict:
    left, right = canonical_rows(before), canonical_rows(after)
    encoded_left = [json.dumps(row, sort_keys=True) for row in left]
    encoded_right = [json.dumps(row, sort_keys=True) for row in right]
    common = sum((Counter(encoded_left) & Counter(encoded_right)).values())
    indexed = []
    for rows in (left, right):
        result = {}
        for row in rows:
            key = (row["frame"], row["track_id"])
            if key in result:
                raise ValueError(f"Duplicate frame/track: {key}")
            result[key] = row
        indexed.append(result)
    a, b = indexed
    changes, examples, pitch_delta = Counter(), [], []
    for key in sorted(a.keys() & b.keys()):
        fields = [field for field in a[key].keys() | b[key].keys()
                  if a[key].get(field) != b[key].get(field)]
        changes.update(fields)
        if a[key].get("bbox_pitch") and b[key].get("bbox_pitch"):
            pitch_delta.extend(abs(float(value) - float(b[key]["bbox_pitch"][field]))
                               for field, value in a[key]["bbox_pitch"].items())
        if fields and len(examples) < 10:
            examples.append({"frame": key[0], "track_id": key[1], "fields": sorted(fields)})
    return {
        "rows_before": len(left), "rows_after": len(right), "exact_common_rows": common,
        "exact_equal": left == right, "missing_frame_tracks": len(a.keys() - b.keys()),
        "added_frame_tracks": len(b.keys() - a.keys()), "changed_fields": dict(changes),
        "examples": examples,
        "pitch_coordinate_absolute_delta_m": {
            "median": float(np.median(pitch_delta)),
            "p90": float(np.quantile(pitch_delta, .9)),
            "p99": float(np.quantile(pitch_delta, .99)),
            "max": max(pitch_delta),
        } if pitch_delta else None,
        "normalized_sha256": [hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                              for rows in (left, right)],
        "ignored_fields": ["id", "image_id", "video_id"],
        "note": "Output equivalence, not an annotated accuracy metric. Track IDs are retained.",
    }


def telemetry(folder: Path) -> dict:
    rows = [json.loads(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()]
    stages = {}
    for stage in sorted({row["stage"] for row in rows}):
        subset = [row for row in rows if row["stage"] == stage and "gpu" in row]
        values = [list(map(float, row["gpu"].split(","))) for row in subset]
        if values:
            stages[stage] = {
                "samples": len(values), "gpu_utilization_sample_mean_pct":
                    statistics.mean(row[0] for row in values),
                "peak_sampled_vram_mib": max(row[1] for row in values),
            }
    return {"stages": stages, "note": "Sampled whole-device utilization, not GPU kernel time"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifests, jobs, predictions = [], [], []
    for folder in (args.before, args.after):
        manifests.append(json.loads((folder / "manifest.json").read_text(encoding="utf-8")))
        jobs.append(json.loads((folder / "core_job.json").read_text(encoding="utf-8")))
        predictions.append(json.loads((folder / "predictions.json").read_text(encoding="utf-8")))
    if any(job["status"] != "done" for job in jobs):
        raise ValueError("Both runs must be complete")
    comparable = {
        key: manifests[0][key] == manifests[1][key]
        for key in ("dataset", "split", "config_hash", "media_sha256")
    }
    comparable["model_files"] = (
        manifests[0]["runtime"]["models"] == manifests[1]["runtime"]["models"]
    )
    comparable["effective_models"] = (
        manifests[0]["effective_models"] == manifests[1]["effective_models"]
    )
    comparable["runtime_environment"] = (
        manifests[0]["runtime"].get("runtime_environment")
        == manifests[1]["runtime"].get("runtime_environment")
    )
    result = {
        "schema_version": "core-comparison.v1", "runs": [str(args.before), str(args.after)],
        "comparable": comparable, "elapsed_s": [job["elapsed"] for job in jobs],
        "speedup": jobs[0]["elapsed"] / jobs[1]["elapsed"],
        "stage_timings": [job["timings"] for job in jobs],
        "predictions": compare_predictions(*predictions),
        "telemetry": [telemetry(folder) for folder in (args.before, args.after)],
        "runtime_environment": [m["runtime"].get("runtime_environment") for m in manifests],
        "limitations": ["Single pair, no confidence interval", "No unified pipeline seed; "
                        "any explicit calibration seed is recorded in runtime_environment",
                        "Host caches/order can affect wall time",
                        "Not full-match or SOTA evidence"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparable": comparable, "elapsed_s": result["elapsed_s"],
                      "speedup": result["speedup"], "predictions": result["predictions"]}))


if __name__ == "__main__":
    main()
