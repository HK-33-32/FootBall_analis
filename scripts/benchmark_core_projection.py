"""Container-only A/B: real fixed homographies/tracks, one matrix load per frame."""

import argparse
import difflib
import hashlib
import json
import shutil
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def optimized_source(original):
    start = '        with open(refined_track_path, "r") as refined_track_file:'
    load = "                homography = np.load(homo_file_path[frame_num - 1])"
    if original.count(start) != 1 or original.count(load) != 1:
        raise ValueError("Unexpected projection source; review before packaging")
    return original.replace(start, "        homography_cache = {}\n" + start).replace(
        load, "                if frame_num not in homography_cache:\n"
        "                    homography_cache[frame_num] = np.load(homo_file_path[frame_num - 1])\n"
        "                homography = homography_cache[frame_num]",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-packaged", action="store_true")
    args = parser.parse_args()
    if args.verify_packaged:
        from IDATR.create_court_file import post_process

        cfg = yaml.safe_load((args.output / "config.yaml").read_text())
        comparison = json.loads((args.output / "comparison.json").read_text())
        source = Path("/app/engine/IDATR/create_court_file.py")
        if text_hash(source.read_text()) != comparison["source_after_sha256"]:
            raise SystemExit("Packaged projection source differs from benchmark candidate")
        started = time.perf_counter()
        post_process(cfg, "test")
        elapsed = time.perf_counter() - started
        outputs = list((args.output / "data/test").glob("*/court_meter_*.txt"))
        if len(outputs) != 1 or digest(outputs[0]) != comparison["samples"][0]["output_sha256"]:
            raise SystemExit("Packaged projection output differs from benchmark")
        result = {"image_id": args.image_id, "source_sha256": text_hash(source.read_text()),
                  "output_sha256": digest(outputs[0]), "wall_time_s": elapsed, "exact_equal": True}
        (args.output / "packaged_verification.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    replay = json.loads((args.source_replay / "replay_manifest.json").read_text())
    if replay.get("status") != "complete":
        raise SystemExit("Requires a completed, immutable calibration replay")
    cfg = yaml.safe_load((args.source_replay / "config.yaml").read_text())
    source_clip = next((args.source_replay / "data/SoccerNetGS/test").iterdir())
    destination = args.output / "data/test" / source_clip.name
    destination.mkdir(parents=True)
    shutil.copy2(source_clip / f"refined_{source_clip.name}.txt", destination)
    shutil.copytree(args.source_replay / "homographies_2", destination / "img1")
    cfg["DATA_DIR"] = str(args.output / "data")
    cfg_path = args.output / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    source = Path("/app/engine/IDATR/create_court_file.py")
    original = source.read_text()
    candidate = optimized_source(original)
    modules = {}
    for name, code in (("baseline", original), ("cached", candidate)):
        namespace = {"__file__": str(source), "__name__": f"benchmark_projection_{name}"}
        exec(compile(code, str(source), "exec"), namespace)
        modules[name] = namespace["post_process"]
    report = {
        "run_id": args.output.name, "timestamp": datetime.now(UTC).isoformat(),
        "dataset": replay["dataset"], "split": "validation", "seed": replay["seed"],
        "image_id": args.image_id, "source_replay_sha256": digest(
            args.source_replay / "replay_manifest.json"),
        "source_before_sha256": text_hash(original), "source_after_sha256": text_hash(candidate),
        "config_sha256": digest(cfg_path),
        "matrix_count": len(list((destination / "img1").glob("*.npy"))),
        "git_commit": replay["source_runtime"]["git_commit"],
        "dirty_worktree": replay["source_runtime"]["dirty_worktree"],
        "models_provenance": "source_replay_sha256 -> source_runtime.runtime.models",
        "software": replay["source_runtime"]["runtime"]["software"],
        "hardware": replay["source_runtime"]["runtime"]["hardware"],
        "note": "Real projection stage only; candidate compiled in benchmark, no model changes",
        "samples": [],
    }
    real_load = np.load
    for repeat in range(args.repeats):
        # Alternate order to reduce consistent first-run cache advantage.
        order = ("baseline", "cached") if repeat % 2 == 0 else ("cached", "baseline")
        for name in order:
            calls = 0

            def counted_load(*values, **kwargs):
                nonlocal calls
                calls += 1
                return real_load(*values, **kwargs)

            np.load = counted_load
            try:
                started = time.perf_counter()
                modules[name](cfg, "test")
                elapsed = time.perf_counter() - started
            finally:
                np.load = real_load
            path = destination / f"court_meter_{source_clip.name}.txt"
            saved = args.output / f"court_{name}_{repeat + 1}.txt"
            shutil.copy2(path, saved)
            sample = {"variant": name, "repeat": repeat + 1, "wall_time_s": elapsed,
                      "matrix_load_calls": calls, "output_sha256": digest(saved)}
            report["samples"].append(sample)
            (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(sample), flush=True)
    report["exact_equal"] = len({s["output_sha256"] for s in report["samples"]}) == 1
    medians = {name: statistics.median(s["wall_time_s"] for s in report["samples"]
                                     if s["variant"] == name) for name in modules}
    report.update(medians_s=medians, speedup=medians["baseline"] / medians["cached"])
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    if not report["exact_equal"] or report["speedup"] <= 1:
        raise SystemExit("Candidate failed acceptance; no production patch exported")
    overlay = args.output / "overlay"
    overlay.mkdir()
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="a/engine/IDATR/create_court_file.py",
        tofile="b/engine/IDATR/create_court_file.py",
    ))
    (overlay / "optimizations.patch").write_text(patch, encoding="utf-8", newline="\n")
    (overlay / "manifest.json").write_text(json.dumps({
        "schema_version": "core-overlay.v1", "patch_sha256": text_hash(patch), "files": {
            "engine/IDATR/create_court_file.py": {
                "before": text_hash(original), "after": text_hash(candidate),
            },
        },
    }, indent=2))
    print(json.dumps({"exact_equal": report["exact_equal"], "speedup": report["speedup"]}))


if __name__ == "__main__":
    main()
