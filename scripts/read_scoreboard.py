"""Read the score off the broadcast graphic, so goals can be confirmed.

A goal is the one event a single camera cannot settle on its own: a ball
passing behind the net looks the same as one crossing the line, and the tracker
usually loses the ball at exactly the moment it matters. The broadcast already
carries the answer in its scoreboard, and the perception container already has
a vision-language model that can read it.

This stages sampled frames on the volume the container mounts, has the model
read each one, and writes a score timeline that `build_match_report.py` accepts
with `--score-timeline`. Readings are then made monotonic: a score never goes
down during a match, so a single misread frame is discarded rather than
inventing a goal and taking it away again.

    python scripts\\read_scoreboard.py --video match.mp4 --output runs\\m\\score.json

The `home`/`away` of the graphic are mapped onto the pitch sides the game state
uses, which is what `--home-side` is for: the broadcast does not know that the
team it lists first is the one defending the left goal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "_vlm_scoreboard_worker.py"
DEFAULT_MODEL = "/opt/weights/checkpoints/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ = "/opt/weights/checkpoints/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"


def sample_frames(video: Path, target: Path, every_s: float, fps: float) -> list[dict]:
    """Write one frame every `every_s` seconds, keeping the match-clock number."""
    import cv2

    target.mkdir(parents=True, exist_ok=True)
    for existing in target.glob("*.png"):
        existing.unlink()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    stride = max(1, int(round(every_s * fps)))
    frames: list[dict] = []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        index += 1
        if (index - 1) % stride:
            continue
        ok, image = capture.retrieve()
        if ok and image is not None:
            path = target / f"{index:06d}.png"
            cv2.imwrite(str(path), image)
            frames.append({"frame": index, "path": path.name})
    capture.release()
    return frames


def monotonic(readings: list[dict], home_side: str) -> list[dict]:
    """Turn raw answers into a timeline that only ever goes up.

    A misread frame is common and a score that falls is impossible, so a
    reading below what has already been established is dropped. A reading that
    jumps by more than one goal at once is accepted -- frames are sampled
    seconds apart and two goals can genuinely fall between them -- but it has
    to be confirmed by the next reading before it counts.
    """
    away_side = "right" if home_side == "left" else "left"
    timeline: list[dict] = []
    best = {"home": 0, "away": 0}
    pending: tuple[int, int, int] | None = None
    for entry in readings:
        if entry.get("home") is None or entry.get("away") is None:
            continue
        home, away, frame = int(entry["home"]), int(entry["away"]), int(entry["frame"])
        if home < best["home"] or away < best["away"]:
            continue  # a score cannot fall: this frame was misread
        if (home, away) != (best["home"], best["away"]):
            if pending is None or pending[:2] != (home, away):
                pending = (home, away, frame)  # wait for a second frame to agree
                continue
            best = {"home": home, "away": away}
            # the goal belongs to the frame the change was first seen, not to
            # the one that confirmed it: samples are seconds apart
            frame = pending[2]
        pending = None
        timeline.append({"frame": frame, home_side: best["home"], away_side: best["away"]})
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--every-s", type=float, default=5.0)
    parser.add_argument(
        "--home-side",
        choices=("left", "right"),
        default="left",
        help="which pitch side the team listed first on the graphic defends",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/perception_benchmark/scoreboard"),
        help="host side of the volume the perception container mounts",
    )
    parser.add_argument("--container-root", default="/data/scoreboard")
    parser.add_argument("--container", default="football-core-benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    frames = sample_frames(args.video, args.export_root / "frames", args.every_s, args.fps)
    manifest = {
        "video": str(args.video),
        "fps": args.fps,
        "every_s": args.every_s,
        "frames": [
            {"frame": item["frame"], "path": f"{args.container_root}/frames/{item['path']}"}
            for item in frames
        ],
    }
    (args.export_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(frames)} frames staged in {args.export_root}")
    if args.stage_only:
        return

    subprocess.run(
        ["docker", "cp", str(WORKER), f"{args.container}:/tmp/{WORKER.name}"], check=True
    )
    subprocess.run(
        [
            "docker", "exec", args.container, "/opt/venv/bin/python", f"/tmp/{WORKER.name}",
            "--manifest", f"{args.container_root}/manifest.json",
            "--output", f"{args.container_root}/readings.json",
            "--model", args.model,
            "--mmproj", args.mmproj,
        ],
        check=True,
    )

    raw = json.loads((args.export_root / "readings.json").read_text(encoding="utf-8"))
    timeline = monotonic(raw["readings"], args.home_side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"readings": timeline, "raw": raw["readings"]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    final = timeline[-1] if timeline else {"left": 0, "right": 0}
    read = sum(1 for item in raw["readings"] if item.get("home") is not None)
    print(
        f"{read}/{len(raw['readings'])} frames readable -> {args.output}; "
        f"final score left {final.get('left', 0)} : right {final.get('right', 0)}"
    )


if __name__ == "__main__":
    sys.exit(main())
