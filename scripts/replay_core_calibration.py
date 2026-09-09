"""Container-only calibration ablation: reuse fixed tracks, preserve every repeat's evidence."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def relocate(value, before, after):
    if isinstance(value, dict):
        return {key: relocate(item, before, after) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, before, after) for item in value]
    if isinstance(value, str) and value.startswith(before + "/"):
        return after + value[len(before):]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit("Output exists; never overwrite a calibration experiment")
    if args.workers < 1 or args.repeats < 1:
        raise SystemExit("workers and repeats must be positive")
    if "FG_CALIB_SEED" not in os.environ:
        raise SystemExit("Choose an explicit FG_CALIB_SEED before the ablation")
    job = json.loads((args.source_job / "job.json").read_text())
    if job["status"] != "done" or job["segments"] != 1:
        raise SystemExit("Requires one completed single-segment job")
    cfg = yaml.safe_load((args.source_job / "config.yaml").read_text())
    original_root = str(Path(cfg["DATA_DIR"]).parents[1])
    cfg = relocate(cfg, original_root, str(args.output))
    if not args.resume:
        shutil.copytree(args.source_job, args.output)
        (args.output / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    clip_dir = Path(cfg["DATA_DIR"]) / "test" / job["clip_name"]
    img_dir = clip_dir / "img1"
    img_dir.mkdir(exist_ok=True)
    scale = {"1080": "1920:1080", "720": "1280:720", "native": None}[job["resolution"]]
    filters = f"fps={job['fps']}" + (f",scale={scale}:flags=bicubic" if scale else "")
    if not args.resume:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-ss", f"{job['start']:.3f}", "-t", f"{job['end'] - job['start']:.3f}",
            "-i", str(args.video), "-vf", filters, "-q:v", "2", "-start_number", "1",
            str(img_dir / "%06d.jpg"),
        ], check=True)
    manifest = {
        "run_id": args.output.name, "timestamp": datetime.now(UTC).isoformat(),
        "dataset": args.sequence, "split": "validation", "image_id": args.image_id,
        "seed": int(os.environ["FG_CALIB_SEED"]), "sampler": "circle-pcg64-v1",
        "workers": args.workers, "frames": len(list(img_dir.glob("*.jpg"))),
        "source_manifest_sha256": digest(args.source_manifest),
        "source_runtime": json.loads(args.source_manifest.read_text()),
        "config_sha256": digest(args.output / "config.yaml"),
        "source_hashes": {name: digest(Path(name)) for name in (
            "/app/engine/kpts.py", "/app/engine/fi_calibration_sampling.py", __file__)},
        "track_sha256": digest(clip_dir / f"refined_{job['clip_name']}.txt"),
        "video_sha256": digest(args.video), "repeats": [],
        "note": "Calibration + projection only; fixed perception/attributes, not end-to-end timing",
    }
    if args.resume:
        previous = json.loads((args.output / "replay_manifest.json").read_text())
        for key in ("seed", "image_id", "workers", "frames", "source_manifest_sha256",
                    "config_sha256", "source_hashes", "track_sha256", "video_sha256"):
            if previous[key] != manifest[key]:
                raise SystemExit(f"Resume fingerprint differs: {key}")
        manifest = previous
    else:
        save(args.output / "replay_manifest.json", manifest)
    sys.path.insert(0, "/app/engine")
    sys.path.insert(0, "/app/engine/IDATR")
    import write_json_file_team as writer
    from create_court_file import post_process

    writer.USE_COLOR_TEAM_OVERRIDE = False
    writer.USE_GK_ANCHORED_SIDES = True
    for repeat in range(len(manifest["repeats"]) + 1, args.repeats + 1):
        started = time.perf_counter()
        log_path = args.output / f"calibration_{repeat}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run([
                sys.executable, "/app/core/runners/run_kpts.py", "--split-dir",
                str(Path(cfg["DATA_DIR"]) / "test"), "--result-dir",
                str(args.output / f"kpts_{repeat}"), "--workers", str(args.workers),
                "--stride", "1",
            ], check=True, stdout=log, stderr=subprocess.STDOUT)
        calibration_s = time.perf_counter() - started
        matrices = args.output / f"homographies_{repeat}"
        matrices.mkdir(exist_ok=args.resume)
        hashes = {}
        for path in sorted(img_dir.glob("*.npy")):
            shutil.copy2(path, matrices / path.name)
            hashes[path.name] = digest(path)
        if len(hashes) != manifest["frames"]:
            raise RuntimeError("Missing homographies; inspect the saved log")
        post_process(cfg, "test")
        writer.main(str(img_dir), str(clip_dir), job["clip_name"])
        predictions = args.output / f"predictions_{repeat}.json"
        shutil.copy2(clip_dir / f"{job['clip_name']}.json", predictions)
        manifest["repeats"].append({"repeat": repeat, "calibration_s": calibration_s,
                                    "homography_hashes": hashes,
                                    "predictions_sha256": digest(predictions)})
        save(args.output / "replay_manifest.json", manifest)
        print(f"repeat={repeat} calibration_s={calibration_s:.2f} matrices={len(hashes)}",
              flush=True)
    manifest["exact_repeat_predictions"] = len({
        item["predictions_sha256"] for item in manifest["repeats"]}) == 1
    manifest["status"] = "complete"
    save(args.output / "replay_manifest.json", manifest)
    print(f"exact_repeat_predictions={manifest['exact_repeat_predictions']}", flush=True)


if __name__ == "__main__":
    main()
