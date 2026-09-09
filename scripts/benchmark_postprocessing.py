"""Reproducible CPU/video A/B benchmark; every output is hashed for equivalence."""

import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path


class SequentialReferenceFrames:
    """Independent all-frame decoder for full-clip pixel/algorithm equivalence.

    It intentionally decodes skipped images as well; not a model/inference stub.
    """

    def __init__(self, path):
        import cv2

        self._capture = cv2.VideoCapture(str(path))
        self.position = 0
        self.last = None

    def __call__(self, number):
        import cv2

        if number < self.position:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.position = 0
        while self.position < number:
            ok, self.last = self._capture.read()
            if not ok:
                return None
            self.position += 1
        return self.last


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--reference-revision", help="load unchanged algorithms from this git revision"
    )
    parser.add_argument("--dataset", default="user-argentina-france-clip")
    parser.add_argument("--split", default="unlabelled-development")
    parser.add_argument("--decoder", choices=["native", "sequential-reference"], default="native")
    parser.add_argument("--min-frame", type=int, default=1)
    parser.add_argument("--max-frame", type=int)
    parser.add_argument(
        "--include-detailed",
        action="store_true",
        help="also time the new report builder; output hash includes the added schema",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    source_hashes = {}
    for name in ("trajectories", "calibration", "gamestate", "match_stats"):
        path = f"src/football_intelligence/{name}.py"
        code = (
            subprocess.check_output(["git", "show", f"{args.reference_revision}:{path}"], cwd=root)
            if args.reference_revision
            else (root / path).read_bytes()
        )
        source_hashes[path] = hashlib.sha256(code).hexdigest()
        if args.reference_revision:
            module = types.ModuleType(f"football_intelligence.{name}")
            module.__package__ = "football_intelligence"
            sys.modules[module.__name__] = module
            exec(compile(code, path, "exec"), module.__dict__)
    from football_intelligence.calibration import CalibrationConfig, drop_uncalibrated
    from football_intelligence.gamestate import RefinementConfig, VideoFrames, refine_predictions
    from football_intelligence.match_stats import StatsConfig, match_statistics
    from football_intelligence.reporting import ReportConfig, detailed_report
    from football_intelligence.trajectories import TrajectoryConfig

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    rows = [
        row
        for row in rows
        if row["frame"] >= args.min_frame
        and (args.max_frame is None or row["frame"] <= args.max_frame)
    ]
    profile = cProfile.Profile()
    timings = []
    hashes = []

    def run():
        if args.video:
            loader = (
                SequentialReferenceFrames(args.video)
                if args.decoder == "sequential-reference"
                else VideoFrames(str(args.video))
            )
            try:
                result, _ = refine_predictions(rows, loader, RefinementConfig(link_tracklets=True))
            finally:
                loader._capture.release()
        else:
            kept, calibration = drop_uncalibrated(rows)
            result = {"statistics": match_statistics(kept), "calibration": calibration}
            if args.include_detailed:
                result["detailed_statistics"] = detailed_report(
                    result["statistics"], predictions=kept
                )
        return result

    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run()
        timings.append(time.perf_counter() - start)
        hashes.append(digest(result))
    # Profile a separate invocation; instrumentation must not distort timings.
    profile.runcall(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(args.output.with_suffix(".prof")))
    config = {
        "refinement": asdict(RefinementConfig(link_tracklets=True)),
        "calibration": asdict(CalibrationConfig()),
        "statistics": asdict(StatsConfig()),
        "trajectories": asdict(TrajectoryConfig()),
    }
    if args.include_detailed:
        config["detailed_report"] = ReportConfig().model_dump()
        path = "src/football_intelligence/reporting.py"
        source_hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    video_hash = None
    if args.video:
        with args.video.open("rb") as video_stream:
            video_hash = hashlib.file_digest(video_stream, "sha256").hexdigest()
    report = {
        "run_id": args.output.stem,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "reference_revision": args.reference_revision,
        "source_sha256": source_hashes,
        "dataset": args.dataset,
        "split": args.split,
        "config": config,
        "config_hash": digest(config),
        "decoder": args.decoder,
        "frame_interval": [args.min_frame, args.max_frame],
        "models": {"inference": "none; deterministic postprocessing of immutable predictions"},
        "prompt_versions": {},
        "schema_versions": {"benchmark": "1.0.0"},
        "seed": 0,
        "hardware": {"processor": platform.processor(), "machine": platform.machine()},
        "software": {name: version(name) for name in ("numpy", "scipy", "opencv-python-headless")},
        "frames_processed": len({row["frame"] for row in rows}),
        "video_sha256": video_hash,
        "input_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "repeats": args.repeats,
        "wall_s": timings,
        "median_s": statistics.median(timings),
        "output_hashes": hashes,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
