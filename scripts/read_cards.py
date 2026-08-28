"""Ask the VLM whether a referee is showing a card, at the moments worth asking.

A foul is not visible in tracking. Its consequence is: play stops, the ball
sits still, and the referee walks over. `match_events.find_stoppages` marks
those moments -- a candidate detector that cannot tell a booking from a
throw-in -- and this settles which of them were cards by looking.

Asking only at candidate frames is what makes it cheap. A ninety-minute match
sampled every five seconds is a thousand questions; asked around the stoppages
a match actually has, it is a few dozen.

**This does not work on every match, and it did not work on the one it was
built against.** At Qatar 2022 the officials wore yellow, and Qwen2.5-VL
answered "yellow" on two of three frames of open play with no card in them --
including on crops tightened to the referee himself. The kit is the yellow it
sees. Treat the output as a suggestion to check by hand, verify it on your own
footage before believing it, and prefer feeding a verified list to
``build_match_report.py --cards`` over trusting this end to end. The stoppage
candidates in the report come from tracking and do not depend on any of this.

    python scripts\\read_cards.py --report runs\\m\\match_report.json ^
      --video match.mp4 --output runs\\m\\cards.json

Then rebuild the report with `--cards runs\\m\\cards.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "_vlm_card_worker.py"
DEFAULT_MODEL = "/opt/weights/checkpoints/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ = "/opt/weights/checkpoints/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"


def frames_to_ask(report: dict, fps: float, spread_s: float, per_stoppage: int) -> list[int]:
    """A few frames around each stoppage: the card is shown after the whistle."""
    wanted: list[int] = []
    for stoppage in report.get("stoppages", []):
        start = int(stoppage["frame"])
        step = max(1, int(round(spread_s * fps / max(1, per_stoppage - 1))))
        for index in range(per_stoppage):
            wanted.append(start + index * step)
    return sorted(set(wanted))


def sample(video: Path, target: Path, frames: list[int]) -> list[dict]:
    import cv2

    target.mkdir(parents=True, exist_ok=True)
    for existing in target.glob("*.png"):
        existing.unlink()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    wanted = set(frames)
    written: list[dict] = []
    index = 0
    while wanted:
        ok = capture.grab()
        if not ok:
            break
        index += 1
        if index not in wanted:
            continue
        wanted.discard(index)
        ok, image = capture.retrieve()
        if ok and image is not None:
            path = target / f"{index:06d}.png"
            cv2.imwrite(str(path), image)
            written.append({"frame": index, "path": path.name})
    capture.release()
    return written


def collapse(readings: list[dict], fps: float, window_s: float = 8.0) -> list[dict]:
    """One card per incident: the same booking shows on several frames."""
    cards: list[dict] = []
    for entry in sorted(readings, key=lambda item: int(item["frame"])):
        colour = entry.get("colour")
        if colour not in ("yellow", "red"):
            continue
        frame = int(entry["frame"])
        if cards and cards[-1]["colour"] == colour and frame - cards[-1]["frame"] <= window_s * fps:
            continue
        cards.append({"frame": frame, "colour": colour, "source": "vlm"})
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--spread-s", type=float, default=4.0)
    parser.add_argument("--frames-per-stoppage", type=int, default=3)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/perception_benchmark/cards"),
        help="host side of the volume the perception container mounts",
    )
    parser.add_argument("--container-root", default="/data/cards")
    parser.add_argument("--container", default="football-core-benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    print(
        "ВНИМАНИЕ: на футболке судьи жёлтого цвета модель отвечает «yellow» и "
        "без карточки. Проверьте ответы вручную, прежде чем им доверять.",
        file=sys.stderr,
    )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    wanted = frames_to_ask(report, args.fps, args.spread_s, args.frames_per_stoppage)
    if not wanted:
        args.output.write_text(json.dumps({"cards": []}, ensure_ascii=False), encoding="utf-8")
        print("no stoppages in the report: nothing to ask about")
        return

    written = sample(args.video, args.export_root / "frames", wanted)
    manifest = {
        "frames": [
            {"frame": item["frame"], "path": f"{args.container_root}/frames/{item['path']}"}
            for item in written
        ]
    }
    (args.export_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(written)} frames around {len(report.get('stoppages', []))} stoppages staged")
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
    cards = collapse(raw["readings"], args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cards": cards, "raw": raw["readings"]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{len(cards)} cards from {len(raw['readings'])} frames -> {args.output}")


if __name__ == "__main__":
    sys.exit(main())
