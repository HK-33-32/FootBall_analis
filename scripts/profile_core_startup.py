"""Container-only, real-input startup profile; writes outside historical jobs."""

import argparse
import cProfile
import functools
import hashlib
import importlib
import json
import os
import platform
import pstats
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--profile-init", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "profiler.py")
    source = json.loads(args.source_manifest.read_text())
    if source["split"] != "development":
        raise SystemExit("Startup development requires development data, not held-out labels")
    if digest(args.video) != source["media_sha256"]:
        raise SystemExit("Source video fingerprint differs")
    manifest = {
        "run_id": args.output.name, "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": source["git_commit"], "dirty_worktree": True,
        "dataset": source["dataset"], "split": "development", "image_id": args.image_id,
        "source_manifest_sha256": digest(args.source_manifest),
        "source_runtime": source, "source_config_sha256": digest(args.config),
        "video_sha256": digest(args.video), "script_sha256": digest(__file__),
        "seed": None, "schema_versions": {"startup_profile": "1.0.0"},
        "prompt_versions": source["prompt_versions"], "stages": {}, "calls": [],
        "status": "preparing", "profile_init": args.profile_init,
        "note": "Isolated tracking process; no concurrent calibration or IDATR. "
                "Nested call times are inclusive and must not be summed.",
    }
    path = args.output / "manifest.json"
    loaded_weights = set()

    def checkpoint():
        save(path, manifest)

    def measure(name, function, *pos, **kw):
        started = time.perf_counter()
        try:
            return function(*pos, **kw)
        finally:
            elapsed = time.perf_counter() - started
            manifest["stages"][name] = elapsed
            checkpoint()
            print(f"[startup] {name}={elapsed:.3f}s", flush=True)

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*pos, **kw):
            if label in {"torch.load", "torch.jit.load"}:
                weight = pos[0] if pos else kw.get("f")
                if not isinstance(weight, str | Path):
                    weight = getattr(weight, "name", None)
                if isinstance(weight, str | Path) and Path(weight).is_file():
                    loaded_weights.add(str(Path(weight).resolve()))
            started = time.perf_counter()
            try:
                return original(*pos, **kw)
            finally:
                manifest["calls"].append({"name": label, "wall_s": time.perf_counter()-started})

        setattr(owner, name, timed)

    checkpoint()
    try:
        import yaml

        cfg = yaml.safe_load(args.config.read_text())
        clip = args.output / "data" / "test" / "ARGFRA-startup"
        frames = clip / "img1"
        frames.mkdir(parents=True)
        cfg.update(DATA_DIR=str(args.output / "data"), IMG_SAVE_DIR=str(args.output / "predicted"),
                   LOG_DIR=str(args.output / "court"))
        effective = args.output / "config.yaml"
        effective.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        manifest["config_hash"] = digest(effective)
        measure("extract", subprocess.run, [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", "0.000",
            "-t", str(source["config"]["end"]), "-i", str(args.video),
            "-vf", "fps=25,scale=1920:1080:flags=bicubic", "-q:v", "2",
            "-start_number", "1", str(frames / "%06d.jpg"),
        ], check=True)
        manifest["frame_hashes"] = {p.name: digest(p) for p in sorted(frames.glob("*.jpg"))}
        os.chdir("/app/engine")
        sys.path.insert(0, "/app/engine")
        module = measure("imports", importlib.import_module, "inference_soccernetGSR")
        torch = module.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; real GPU startup benchmark cannot proceed")
        manifest["hardware"] = {"gpu": torch.cuda.get_device_name(0),
                                "cpu_count": os.cpu_count(),
                                "torch_threads": torch.get_num_threads()}
        manifest["software"] = {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda}
        manifest["source_hashes"] = {str(p): digest(p) for p in (
            Path(module.__file__), Path("detectors.py"), Path("jersey_model/CLIPFinetune.py"),
            Path("/opt/venv/lib/python3.12/site-packages/rfdetr/utilities/files.py"))}
        manifest["models"] = source.get("effective_models", source.get("models", {}))
        wrap(module, "build_detector", "detector_constructor")
        wrap(module, "FeatureExtractor", "reid_constructor")
        # Keep the class intact: direct-checkpoint loading is a real classmethod.
        wrap(module.CLIPFinetune, "__init__", "clip_constructor")
        wrap(torch, "load", "torch.load")
        wrap(torch.jit, "load", "torch.jit.load")
        wrap(module.clip, "load", "clip.load")
        wrap(torch.nn.Module, "load_state_dict", "load_state_dict")
        device = torch.device("cuda")
        manifest["status"] = "initializing"
        checkpoint()
        profiler = cProfile.Profile()
        if args.profile_init:
            profiler.enable()
        pipeline = measure("initialization", module.GSRPipeline, cfg, device)
        if args.profile_init:
            profiler.disable()
            profiler.dump_stats(str(args.output / "initialization.prof"))
            with (args.output / "initialization_top.txt").open("w") as stream:
                stats = pstats.Stats(profiler, stream=stream).strip_dirs()
                stats.sort_stats("cumulative").print_stats(45)
        torch.cuda.synchronize()
        manifest["status"] = "tracking"
        checkpoint()
        pipeline.spent = {}
        measure("tracking", pipeline.run, ["test"])
        torch.cuda.synchronize()
        manifest["tracking_timers"] = pipeline.spent
        manifest["outputs"] = {str(p.relative_to(args.output)): digest(p)
                               for p in clip.glob("interpolate_*.txt")}
        manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        manifest["loaded_weight_hashes"] = {p: digest(p) for p in sorted(loaded_weights)}
        manifest["status"] = "complete"
        checkpoint()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        checkpoint()
        raise


if __name__ == "__main__":
    main()
