"""Re-read jersey numbers from the best crops of each stitched identity.

The perception backend asks its VLM about eight near-consecutive crops of each
*tracklet*. This script pools the crops of a whole linked identity, picks views
that differ from one another, and asks the same VLM again — one question set
per player instead of one per fragment.

The reader itself has to run inside the Football Core container, which is where
the GGUF weights live, so the crops are staged on the volume both sides share
and ``scripts/_vlm_jersey_worker.py`` is executed there over ``docker exec``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from football_intelligence.gamestate import (  # noqa: E402
    RefinementConfig,
    assign_teams,
    link_tracklets,
    load_reid_detections,
    load_reid_embeddings,
    sequence_frame_loader,
    summarise_tracks,
)
from football_intelligence.jersey_views import (  # noqa: E402
    JerseyViewConfig,
    collect_identity_views,
    views_manifest,
)

WORKER = Path(__file__).with_name("_vlm_jersey_worker.py")


def stage_views(
    predictions: list[dict],
    sequence: str,
    frames_root: Path,
    reid_pickle: Path | None,
    export_dir: Path,
    view_config: JerseyViewConfig,
    refine_config: RefinementConfig,
) -> dict:
    """Write the chosen crops and a manifest describing them."""
    frames = sequence_frame_loader(str(frames_root), sequence)
    summaries = summarise_tracks(predictions, frames, refine_config)
    assign_teams(summaries, predictions, refine_config)
    embeddings = load_reid_embeddings(str(reid_pickle), predictions) if reid_pickle else None
    groups, link_report = link_tracklets(summaries, refine_config, embeddings)
    per_detection = load_reid_detections(str(reid_pickle)) if reid_pickle else None

    frames = sequence_frame_loader(str(frames_root), sequence)
    selected, crops = collect_identity_views(
        predictions, frames, groups, per_detection, view_config
    )

    crop_dir = export_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for existing in crop_dir.glob("*.png"):
        existing.unlink()
    for observation_id, crop in crops.items():
        cv2.imwrite(str(export_dir / f"crops/{observation_id.replace(':', '_')}.png"), crop)

    manifest = views_manifest(selected, sequence, view_config)
    manifest["linking"] = link_report
    manifest["tracks"] = len(summaries)
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_worker(container: str, container_dir: str, group_size: int, crops: int) -> None:
    subprocess.run(["docker", "cp", str(WORKER), f"{container}:/tmp/{WORKER.name}"], check=True)
    subprocess.run(
        [
            "docker",
            "exec",
            container,
            "/opt/venv/bin/python",
            f"/tmp/{WORKER.name}",
            "--manifest",
            f"{container_dir}/manifest.json",
            "--output",
            f"{container_dir}/reads.json",
            "--group-size",
            str(group_size),
            "--crops-per-identity",
            str(crops),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-template", default="soccernet_valid_{sequence}_core")
    parser.add_argument("--frames-root", type=Path, default=Path("data/soccernet/valid"))
    parser.add_argument("--jobs-root", type=Path, default=Path("data/perception_benchmark/jobs"))
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/perception_benchmark/jersey_reads"),
        help="host side of the volume the perception container mounts",
    )
    parser.add_argument(
        "--container-root",
        default="/data/jersey_reads",
        help="same directory as --export-root, as the container sees it",
    )
    parser.add_argument("--container", default="football-core-benchmark")
    parser.add_argument("--views-per-identity", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--min-time-gap-ms", type=int, default=400)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    view_config = JerseyViewConfig(
        views_per_identity=args.views_per_identity, min_time_gap_ms=args.min_time_gap_ms
    )
    refine_config = RefinementConfig(link_tracklets=True)

    for sequence in args.sequences:
        run_dir = args.runs_root / args.run_template.format(sequence=sequence)
        if not (run_dir / "predictions.json").is_file():
            legacy = args.runs_root / args.run_template.format(
                sequence=sequence.replace("-", "").lower()
            )
            if (legacy / "predictions.json").is_file():
                run_dir = legacy
        source = run_dir / "predictions.json"
        if not source.is_file():
            print(f"skip {sequence}: {source} missing", file=sys.stderr)
            continue
        predictions = json.loads(source.read_text(encoding="utf-8"))["predictions"]

        job = json.loads((run_dir / "core_job.json").read_text(encoding="utf-8")).get("id")
        pickles = sorted((args.jobs_root / str(job)).rglob("*.pkl")) if job else []
        reid_pickle = pickles[0] if pickles else None

        export_dir = args.export_root / sequence
        started = time.perf_counter()
        manifest = stage_views(
            predictions,
            sequence,
            args.frames_root,
            reid_pickle,
            export_dir,
            view_config,
            refine_config,
        )
        crops = sum(len(entry["views"]) for entry in manifest["identities"])
        print(
            f"{sequence}: {len(manifest['identities'])} identities, {crops} crops staged "
            f"in {time.perf_counter() - started:.1f}s -> {export_dir}",
            flush=True,
        )
        if args.stage_only:
            continue
        run_worker(
            args.container,
            f"{args.container_root}/{sequence}",
            args.group_size,
            args.views_per_identity,
        )
        reads = json.loads((export_dir / "reads.json").read_text(encoding="utf-8"))
        target = run_dir / "jersey_reads.json"
        target.write_text(json.dumps(reads, ensure_ascii=False, indent=2), encoding="utf-8")
        resolved = sum(1 for entry in reads["identities"] if entry["tally"])
        print(
            f"{sequence}: {resolved}/{len(reads['identities'])} identities read "
            f"in {reads['wall_time_s']}s -> {target}",
            flush=True,
        )


if __name__ == "__main__":
    main()
