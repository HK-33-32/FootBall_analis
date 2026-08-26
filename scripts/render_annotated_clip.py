"""Render a SoccerNet clip with its game state drawn on top.

Two broadcast panels side by side -- by default the perception output as
delivered and the refined output -- over a shared pitch minimap that carries
the ground truth as hollow markers. That layout makes the thing GS-HOTA
actually measures visible: whether a box carries the right team colour and the
right number, and whether the point lands where the annotation says it does.

Ground truth is read for display only; nothing here feeds the pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

TEAM_COLOURS = {"left": (235, 160, 40), "right": (60, 90, 235)}
REFEREE_COLOUR = (60, 220, 240)
BALL_COLOUR = (250, 250, 250)
UNKNOWN_COLOUR = (170, 170, 170)
LINE_COLOUR = (140, 190, 140)
PANEL_BACKGROUND = (28, 32, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_of(attributes: dict[str, Any]) -> tuple[int, int, int]:
    role = attributes.get("role")
    if role == "ball":
        return BALL_COLOUR
    if role == "referee":
        return REFEREE_COLOUR
    return TEAM_COLOURS.get(attributes.get("team"), UNKNOWN_COLOUR)


def label_of(attributes: dict[str, Any]) -> str:
    role = attributes.get("role")
    if role == "referee":
        return "REF"
    if role == "ball":
        return ""
    jersey = attributes.get("jersey")
    number = str(jersey) if jersey not in (None, "") else "?"
    return f"GK {number}" if role == "goalkeeper" else number


def by_frame(detections: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for detection in detections:
        grouped[int(detection["frame"])].append(detection)
    return grouped


def load_predictions(path: Path) -> dict[int, list[dict]]:
    return by_frame(json.loads(path.read_text(encoding="utf-8"))["predictions"])


def load_ground_truth(sequence_dir: Path) -> dict[int, list[dict]]:
    truth = json.loads((sequence_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    frames = {image["image_id"]: int(str(image["image_id"])[-6:]) for image in truth["images"]}
    rows = []
    for annotation in truth["annotations"]:
        if not annotation.get("bbox_pitch"):
            continue
        rows.append({**annotation, "frame": frames[annotation["image_id"]]})
    return by_frame(rows)


def draw_broadcast(image: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    canvas = image.copy()
    for detection in sorted(detections, key=lambda item: item["bbox_image"]["y"]):
        attributes = detection.get("attributes") or {}
        box = detection["bbox_image"]
        x0, y0 = int(box["x"]), int(box["y"])
        x1, y1 = int(box["x"] + box["w"]), int(box["y"] + box["h"])
        colour = colour_of(attributes)
        if attributes.get("role") == "ball":
            cv2.circle(canvas, ((x0 + x1) // 2, (y0 + y1) // 2), 9, colour, 2)
            continue
        thickness = 3 if attributes.get("role") == "goalkeeper" else 2
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)
        text = label_of(attributes)
        if not text:
            continue
        (width, height), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        cv2.rectangle(canvas, (x0, y0 - height - 8), (x0 + width + 8, y0), colour, -1)
        cv2.putText(canvas, text, (x0 + 4, y0 - 5), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (24, 24, 24), -1)
    cv2.putText(canvas, title, (16, 32), FONT, 0.95, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def pitch_canvas(width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    canvas = np.full((height, width, 3), PANEL_BACKGROUND, dtype=np.uint8)
    # leave a column on the left for the legend, and a little air top and bottom
    scale = min((width - 380) / PITCH_LENGTH_M, (height - 50) / PITCH_WIDTH_M)
    offset_x = int(width - PITCH_LENGTH_M * scale - 40)
    offset_y = int((height - PITCH_WIDTH_M * scale) / 2)

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            int(offset_x + (x + PITCH_LENGTH_M / 2) * scale),
            int(offset_y + (y + PITCH_WIDTH_M / 2) * scale),
        )

    cv2.rectangle(canvas, point(-52.5, -34), point(52.5, 34), LINE_COLOUR, 2)
    cv2.line(canvas, point(0, -34), point(0, 34), LINE_COLOUR, 2)
    cv2.circle(canvas, point(0, 0), int(9.15 * scale), LINE_COLOUR, 2)
    for side in (-1, 1):
        cv2.rectangle(
            canvas,
            point(side * 52.5, -20.16),
            point(side * (52.5 - 16.5), 20.16),
            LINE_COLOUR,
            2,
        )
        cv2.rectangle(
            canvas,
            point(side * 52.5, -9.16),
            point(side * (52.5 - 5.5), 9.16),
            LINE_COLOUR,
            2,
        )
    return canvas, scale, offset_x, offset_y


def draw_minimap(
    width: int,
    height: int,
    predicted: list[dict],
    truth: list[dict],
    frame_number: int,
    sequence: str,
    with_truth: bool = True,
) -> np.ndarray:
    canvas, scale, offset_x, offset_y = pitch_canvas(width, height)

    def place(detection: dict) -> tuple[int, int] | None:
        pitch = detection.get("bbox_pitch") or {}
        if "x_bottom_middle" not in pitch:
            return None
        return (
            int(offset_x + (float(pitch["x_bottom_middle"]) + PITCH_LENGTH_M / 2) * scale),
            int(offset_y + (float(pitch["y_bottom_middle"]) + PITCH_WIDTH_M / 2) * scale),
        )

    for detection in truth:
        centre = place(detection)
        if centre is None:
            continue
        cv2.circle(canvas, centre, 11, colour_of(detection.get("attributes") or {}), 1)

    for detection in predicted:
        centre = place(detection)
        if centre is None:
            continue
        attributes = detection.get("attributes") or {}
        colour = colour_of(attributes)
        radius = 4 if attributes.get("role") == "ball" else 7
        cv2.circle(canvas, centre, radius, colour, -1)
        text = label_of(attributes)
        if text and text != "REF":
            cv2.putText(
                canvas,
                text,
                (centre[0] + 9, centre[1] + 5),
                FONT,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )

    _draw_legend(canvas, sequence, frame_number, offset_x, with_truth)
    return canvas


def _draw_legend(
    canvas: np.ndarray,
    sequence: str,
    frame_number: int,
    margin: int,
    with_truth: bool = True,
) -> None:
    """Fill the space the pitch leaves over with a key to the markers."""
    x = 26
    if margin < 240:  # no room beside the pitch, fall back to one caption line
        cv2.putText(
            canvas,
            f"{sequence}  frame {frame_number:04d}"
            + ("   filled = prediction, hollow = ground truth" if with_truth else ""),
            (x, canvas.shape[0] - 16),
            FONT,
            0.6,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        return

    y = 48
    cv2.putText(canvas, sequence, (x, y), FONT, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, f"frame {frame_number:04d}", (x, y + 30), FONT, 0.6, (185, 185, 185), 1, cv2.LINE_AA
    )
    rows: list[tuple[str, tuple[int, int, int], bool]] = [
        ("prediction", (215, 215, 215), True),
        *([("ground truth", (215, 215, 215), False)] if with_truth else []),
        ("team left", TEAM_COLOURS["left"], True),
        ("team right", TEAM_COLOURS["right"], True),
        ("referee", REFEREE_COLOUR, True),
        ("no team decided", UNKNOWN_COLOUR, True),
    ]
    y += 68
    for text, colour, filled in rows:
        cv2.circle(canvas, (x + 8, y - 5), 7, colour, -1 if filled else 1)
        cv2.putText(canvas, text, (x + 28, y), FONT, 0.55, (205, 205, 205), 1, cv2.LINE_AA)
        y += 30
    cv2.putText(
        canvas,
        'label = shirt number ("?" unread)',
        (x, y + 12),
        FONT,
        0.5,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--predictions", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--titles", nargs="+", default=None, help="one per predictions file, in the same order"
    )
    parser.add_argument("--minimap-from", type=int, default=-1, help="index of the panel to plot")
    parser.add_argument("--frames-root", type=Path, default=Path("data/soccernet/valid"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--no-ground-truth", action="store_true")
    args = parser.parse_args()

    sequence_dir = args.frames_root / args.sequence
    panels = [load_predictions(path) for path in args.predictions]
    titles = args.titles or [path.stem for path in args.predictions]
    if len(titles) != len(panels):
        raise SystemExit("--titles must give one title per predictions file")
    truth = {} if args.no_ground_truth else load_ground_truth(sequence_dir)

    image_dir = sequence_dir / "img1"
    available = sorted(int(path.stem) for path in image_dir.glob("*.jpg"))
    if not available:
        raise SystemExit(f"no extracted frames in {image_dir}")
    last = args.end or available[-1]
    numbers = [number for number in available if args.start <= number <= last]

    def even(value: float) -> int:
        # H.264 with 4:2:0 chroma needs both dimensions even
        return int(value) // 2 * 2

    first = cv2.imread(str(image_dir / f"{numbers[0]:06d}.jpg"))
    panel_width = even(args.panel_width)
    panel_height = even(first.shape[0] * panel_width / first.shape[1])
    total_width = panel_width * len(panels)
    minimap_height = even(total_width * PITCH_WIDTH_M / PITCH_LENGTH_M / 2.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{total_width}x{panel_height + minimap_height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    for number in numbers:
        image = cv2.imread(str(image_dir / f"{number:06d}.jpg"))
        if image is None:
            continue
        drawn = [
            cv2.resize(
                draw_broadcast(image, panel.get(number, []), title),
                (panel_width, panel_height),
                interpolation=cv2.INTER_AREA,
            )
            for panel, title in zip(panels, titles, strict=True)
        ]
        minimap = draw_minimap(
            total_width,
            minimap_height,
            panels[args.minimap_from].get(number, []),
            truth.get(number, []),
            number,
            args.sequence,
            not args.no_ground_truth,
        )
        encoder.stdin.write(np.vstack([np.hstack(drawn), minimap]).tobytes())

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed")
    size_mb = args.output.stat().st_size / 1e6
    print(f"{args.sequence}: {len(numbers)} frames -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
