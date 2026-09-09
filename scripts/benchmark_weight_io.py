"""Real RF-DETR weight hashing A/B; export a source-checked patch after acceptance."""

import argparse
import ast
import difflib
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def optimized_source(original, block_bytes):
    if not isinstance(block_bytes, int) or isinstance(block_bytes, bool) or block_bytes < 4096:
        raise ValueError("Read block must be an integer of at least 4096 bytes")
    needle = 'for chunk in iter(lambda: f.read(4096), b""):'
    if original.count(needle) != 1:
        raise ValueError("Unexpected upstream MD5 source; review before changing")
    return original.replace(needle, f'for chunk in iter(lambda: f.read({block_bytes}), b""):')


def load_function(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_compute_file_md5")
    scope = {"hashlib": hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<upstream-md5>", "exec"), scope)
    return scope["_compute_file_md5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.source.read_text()
    cfg_text = args.config.read_text()
    cfg = json.loads(cfg_text)
    candidate = optimized_source(original, cfg["rfdetr_md5_read_bytes"])
    before, after = load_function(original), load_function(candidate)
    parent = json.loads(args.source_manifest.read_text())
    result = {
        "run_id": args.output.name, "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": parent["git_commit"], "dirty_worktree": True,
        "dataset": "locally cached RF-DETR weights; no annotations", "split": "development",
        "image_id": args.image_id, "config": cfg, "config_hash": text_hash(cfg_text),
        "source_before_sha256": text_hash(original), "source_after_sha256": text_hash(candidate),
        "source_manifest_sha256": text_hash(args.source_manifest.read_text()),
        "models": parent["loaded_weight_hashes"], "hardware": parent["hardware"],
        "software": parent["software"], "seed": None, "prompt_versions": {},
        "schema_versions": {"weight_io": "1.0.0"}, "samples": [],
    }
    for repeat in range(3):
        order = [("baseline", before), ("candidate", after)]
        for label, function in order if repeat % 2 == 0 else reversed(order):
            started = time.perf_counter()
            digest = function(str(args.weight))
            elapsed = time.perf_counter() - started
            result["samples"].append({"repeat": repeat, "variant": label,
                                      "wall_s": elapsed, "md5": digest})
            (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
            print(f"{label} repeat={repeat} seconds={elapsed:.3f} md5={digest}", flush=True)
    medians = {name: statistics.median(row["wall_s"] for row in result["samples"]
                                      if row["variant"] == name)
               for name in ("baseline", "candidate")}
    result["median_s"] = medians
    result["speedup"] = medians["baseline"] / medians["candidate"]
    result["exact_equal"] = len({row["md5"] for row in result["samples"]}) == 1
    result["status"] = "complete"
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    if not result["exact_equal"] or result["speedup"] <= 1:
        raise SystemExit("Acceptance failed; no production overlay exported")
    overlay = args.output / "overlay"
    overlay.mkdir()
    relative = "utilities/files.py"
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile=f"a/{relative}", tofile=f"b/{relative}",
    ))
    (overlay / "optimizations.patch").write_text(patch)
    (overlay / "manifest.json").write_text(json.dumps({
        "schema_version": "core-overlay.v1", "base_image_id": args.image_id,
        "patch_sha256": text_hash(patch), "config": cfg,
        "files": {relative: {"before": text_hash(original), "after": text_hash(candidate)}},
    }, indent=2))


if __name__ == "__main__":
    main()
