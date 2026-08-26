"""Run one SoccerNet validation sequence through Football Core and export artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-url", default="http://localhost:8000")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    args = parser.parse_args()
    base_url = args.core_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    response = httpx.post(
        f"{base_url}/v1/videos/register",
        data={"path": args.media_path, "name": args.sequence},
        timeout=60,
    )
    response.raise_for_status()
    video = response.json()
    payload = {
        "video_id": video["id"],
        "start": 0,
        "end": args.duration,
        "fps": args.fps,
        "resolution": "1080",
        "detector": "rfdetr",
        "detector_size": "large",
        "jersey_mode": "CLIP",
        "jersey_reader": "qwen-vl",
        "jersey_stride": 3,
        "calib_stride": 1,
        "render_video": args.render,
        "analytics": args.analytics,
    }
    response = httpx.post(f"{base_url}/v1/jobs", json=payload, timeout=60)
    response.raise_for_status()
    job_id = response.json()["id"]
    last_stage = None
    while True:
        response = httpx.get(f"{base_url}/v1/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        job = response.json()
        if job.get("stage") != last_stage:
            print(f"{job['stage']}: {float(job.get('progress', 0)):.1%}", flush=True)
            last_stage = job.get("stage")
        if job["status"] == "done":
            break
        if job["status"] in {"error", "canceled"}:
            raise RuntimeError(f"job {job_id}: {job['status']}: {job.get('error', '')}")
        time.sleep(5)

    (args.output_dir / "core_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    endpoints = {"predictions.json": "predictions"}
    if args.analytics:
        endpoints.update(
            {
                "events.json": "events",
                "player_stats.json": "players",
            }
        )
    if args.render:
        endpoints["annotated.mp4"] = "video"
    for filename, endpoint in endpoints.items():
        with httpx.stream("GET", f"{base_url}/v1/jobs/{job_id}/{endpoint}", timeout=None) as res:
            res.raise_for_status()
            with (args.output_dir / filename).open("wb") as output:
                for chunk in res.iter_bytes(1024 * 1024):
                    output.write(chunk)
    print(f"done: {job_id} -> {args.output_dir}")


if __name__ == "__main__":
    main()
