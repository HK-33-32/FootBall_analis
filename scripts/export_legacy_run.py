"""Export a reproducible Football Core run with hashes and a compact report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    source_video = args.source_video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "annotated.mp4": "result.mp4",
        "events.json": "events.json",
        "player_stats.json": "player_stats.json",
        "player_stats.csv": "player_stats.csv",
        "predictions.json": "predictions.json",
        "effective_config.yaml": "config.yaml",
        "core_job.json": "job.json",
    }
    for target_name, source_name in required.items():
        source = job_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / target_name)

    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    events_doc = json.loads((job_dir / "events.json").read_text(encoding="utf-8"))
    stats = json.loads((job_dir / "player_stats.json").read_text(encoding="utf-8"))
    event_counts = Counter(event["type"] for event in events_doc["events"])
    copied = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(),
        "source_video": {
            "path": str(source_video),
            "bytes": source_video.stat().st_size,
            "sha256": sha256(source_video),
        },
        "source_job": str(job_dir),
        "job_id": job["id"],
        "status": job["status"],
        "effective_settings": {
            key: job[key]
            for key in (
                "start",
                "end",
                "fps",
                "resolution",
                "detector",
                "detector_size",
                "jersey_reader",
                "render_video",
                "analytics",
            )
        },
        "compute": {"elapsed_seconds": job["elapsed"], "timings": job["timings"]},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in copied
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    identification = stats["identification"]
    team_lines = []
    for team, values in stats["teams"].items():
        team_lines.append(
            f"| {team} | {values['possession_s']:.2f} s | "
            f"{values['possession_share']:.1%} | {values['touches']} | "
            f"{values['passes']} | {values['shots']} |"
        )
    event_lines = "\n".join(
        f"- `{event_type}`: {count}" for event_type, count in event_counts.most_common()
    )
    meta = stats["meta"]
    ball = meta["ball_coverage"]
    frame_summary = (
        f"{meta['frames_total']}; detections: {meta['frames_with_detections']}; "
        f"ball: {meta['frames_with_ball']}"
    )
    ball_summary = (
        f"{ball['share']:.1%} ({ball['observed']} observed + "
        f"{ball['interpolated']} interpolated frames)"
    )
    report = f"""# ARG–FRA 2022 — annotated clip report

Source SHA-256: `{manifest['source_video']['sha256']}`

The run is a model inference result, not ground truth. Passes and possession are
heuristic ledger events until they are validated against human annotations.

## Run

- Interval: {job['start']:.2f}–{job['end']:.2f} s
- Sampling: {job['fps']} FPS, {job['resolution']}p
- Detector: {job['detector']} {job['detector_size']}
- Jersey reader: {job['jersey_reader']}
- Wall time: {job['elapsed']:.1f} s
- Frames: {frame_summary}
- Ball coverage: {ball_summary}
- Identified players: {identification['identified_players']}; anonymous fragments:
  {identification['anonymous_fragments']}
- On-ball attribution coverage: {identification['attributed_share']:.1%}

## Team-level estimates

| Camera-side team | Possession | Share | Touches | Passes | Shots |
|---|---:|---:|---:|---:|---:|
{chr(10).join(team_lines)}

## Event ledger ({len(events_doc['events'])})

{event_lines}

See `player_stats.json`/`.csv` for player-level estimates and `events.json` for
the full timeline. `annotated.mp4` contains the rendered detections, tracks,
jersey hypotheses and pitch projection.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
