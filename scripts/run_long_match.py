"""Run a full-length match: screen it, cut it into jobs, and put the pieces back.

Every step writes to disk before the next begins, so an interrupted run picks
up where it stopped instead of starting the day again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from football_intelligence.longmatch import (  # noqa: E402
    ChunkConfig,
    merge_chunks,
    plan_chunks,
)
from football_intelligence.playability import PlayabilityConfig, screen  # noqa: E402

# Measured against the football-core backend after its per-frame model calls
# were batched and its crop preparation spread over the cores: 7.6 s of football
# in 180 s and a 30 s SoccerNet clip in 427 s, so 24x and 14x real time. The
# higher figure is the one used, being the broadcast footage this runs on.
# This is wall time, not GPU work -- the card still has capacity to spare.
REALTIME_FACTOR = 24


def transcode(source: Path, target: Path, start: float, duration: float, fps: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-an", "-vf", f"scale=1920:1080:flags=lanczos,fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            str(target),
        ],
        check=True,
    )


def run_core(base_url: str, media_path: str, name: str, fps: int, duration: float) -> dict:
    response = httpx.post(
        f"{base_url}/v1/videos/register", data={"path": media_path, "name": name}, timeout=60
    )
    response.raise_for_status()
    video = response.json()
    payload = {
        "video_id": video["id"], "start": 0, "end": duration, "fps": fps,
        "resolution": "1080", "detector": "rfdetr", "detector_size": "large",
        "jersey_mode": "CLIP", "jersey_reader": "qwen-vl", "jersey_stride": 3,
        "calib_stride": 1, "render_video": False, "analytics": False,
    }
    response = httpx.post(f"{base_url}/v1/jobs", json=payload, timeout=60)
    response.raise_for_status()
    job_id = response.json()["id"]
    while True:
        response = httpx.get(f"{base_url}/v1/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        job = response.json()
        if job["status"] == "done":
            return job
        if job["status"] in {"error", "canceled"}:
            raise RuntimeError(f"job {job_id}: {job['status']}: {job.get('error', '')}")
        time.sleep(5)


def fetch_predictions(base_url: str, job_id: str, target: Path) -> None:
    with httpx.stream("GET", f"{base_url}/v1/jobs/{job_id}/predictions", timeout=None) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for block in response.iter_bytes(1024 * 1024):
                handle.write(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--media-root", type=Path, default=Path("data/perception_benchmark/media"))
    parser.add_argument("--container-media-root", default="/data/media")
    parser.add_argument("--core-url", default="http://localhost:8000")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--chunk-s", type=float, default=60.0)
    parser.add_argument("--min-grass-share", type=float, default=0.40)
    parser.add_argument("--plan-only", action="store_true", help="screen and plan, run nothing")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_file = args.output_dir / "plan.json"

    if plan_file.is_file():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        print(f"reusing the plan in {plan_file}")
    else:
        started = time.perf_counter()
        config = PlayabilityConfig(min_grass_share=args.min_grass_share)
        scan = screen(str(args.video), config)
        chunks = plan_chunks(scan["segments"], ChunkConfig(max_chunk_s=args.chunk_s, fps=args.fps))
        plan = {"screen": scan, "chunks": chunks, "scan_s": round(time.perf_counter() - started, 1)}
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    scan = plan["screen"]
    chunks = plan["chunks"]
    machine_hours = scan["playable_s"] * REALTIME_FACTOR / 3600
    print(
        f"{scan['duration_s']:.0f}s of video screened in {plan.get('scan_s', 0)}s: "
        f"{scan['playable_s']:.0f}s playable ({scan['playable_share']:.0%}) in "
        f"{len(scan['segments'])} stretches, {len(chunks)} chunks. "
        f"Skipping {scan['skipped_s']:.0f}s saves about "
        f"{scan['skipped_s'] * REALTIME_FACTOR / 3600:.1f} machine-hours; the rest is "
        f"about {machine_hours:.1f}."
    )
    if args.plan_only:
        return

    parts = []
    for chunk in chunks:
        chunk_dir = args.output_dir / chunk["name"]
        chunk_dir.mkdir(parents=True, exist_ok=True)
        predictions_file = chunk_dir / "predictions.json"
        if predictions_file.is_file():
            print(f"{chunk['name']}: already done, skipping")
        else:
            clip_name = f"{args.name}-{chunk['name']}"
            media = args.media_root / clip_name / f"{clip_name}.mp4"
            if not media.is_file():
                transcode(args.video, media, chunk["start_s"], chunk["duration_s"], args.fps)
            started = time.perf_counter()
            job = run_core(
                args.core_url,
                f"{args.container_media_root}/{clip_name}/{clip_name}.mp4",
                clip_name,
                args.fps,
                chunk["duration_s"],
            )
            fetch_predictions(args.core_url, job["id"], predictions_file)
            (chunk_dir / "core_job.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"{chunk['name']}: {chunk['duration_s']:.1f}s of football in "
                f"{time.perf_counter() - started:.0f}s"
            )
        payload = json.loads(predictions_file.read_text(encoding="utf-8"))
        parts.append((chunk, payload["predictions"]))

    merged = merge_chunks(parts, ChunkConfig(fps=args.fps))
    target = args.output_dir / "predictions.json"
    target.write_text(
        json.dumps({"predictions": merged["predictions"]}, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "chunks.json").write_text(
        json.dumps(
            {"chunks": merged["chunks"], "identities": merged["identities"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"merged {len(parts)} chunks: {merged['detections']} detections, "
        f"{merged['identities']} identities -> {target}"
    )


if __name__ == "__main__":
    main()
