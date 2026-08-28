"""Finding the parts of a broadcast worth sending to the GPU.

Perception costs about sixty-five times real time on this hardware, so a
ninety-minute match is the better part of a working week of wall time. Much of that is spent on
footage that cannot produce a game state at all: replays, dugout reactions,
crowd shots, graphics, and the close-ups a director cuts to whenever the ball
is dead. Measured on broadcast footage of the 2022 final, 47% of the clip is
of that kind.

Deciding which is which does not need a network. A camera showing the pitch
shows a great deal of grass; a close-up of a player's face does not. Scanning
the decoded video at a stride and thresholding the green share separates the
two at 99% recall -- meaning almost nothing that would have produced a game
state is thrown away -- while discarding roughly half the footage. Precision is
deliberately the loose end: this pass is meant to be permissive, and
:mod:`football_intelligence.calibration` removes what survives it but still
cannot be trusted.

The output is a list of time ranges to feed the perception backend, so the cost
of a match falls in proportion to how much of it is actually football.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlayabilityConfig:
    """How the scan samples the video and where it draws the line."""

    stride_frames: int = 5
    width: int = 320
    height: int = 180
    grass_hue: tuple[int, int] = (32, 90)
    grass_min_saturation: int = 60
    grass_min_value: int = 40
    min_grass_share: float = 0.40
    # Hysteresis: a stretch has to be clearly playable to open, and clearly
    # not to close, so a single dark sample does not chop a passage in two.
    open_grass_share: float = 0.45
    min_segment_s: float = 1.0
    pad_s: float = 0.4
    merge_gap_s: float = 1.0


def grass_share(frame: np.ndarray, config: PlayabilityConfig) -> float:
    """Fraction of the frame that looks like a lit pitch."""
    import cv2

    small = cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    low, high = config.grass_hue
    grass = (
        (hue >= low)
        & (hue <= high)
        & (saturation > config.grass_min_saturation)
        & (value > config.grass_min_value)
    )
    return float(grass.mean())


def scan_video(
    path: str, config: PlayabilityConfig | None = None, fps: float | None = None
) -> dict[str, Any]:
    """Sample the video and score every sampled frame for grass."""
    import cv2

    config = config or PlayabilityConfig()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(path)
    rate = fps or capture.get(cv2.CAP_PROP_FPS) or 25.0
    samples: list[tuple[int, float]] = []
    index = 0
    while True:
        ok = capture.grab()  # decode only the frames we intend to look at
        if not ok:
            break
        if index % config.stride_frames == 0:
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                samples.append((index, grass_share(frame, config)))
        index += 1
    capture.release()
    return {"fps": rate, "frames": index, "samples": samples, "stride": config.stride_frames}


def playable_segments(scan: dict[str, Any], config: PlayabilityConfig | None = None) -> list[dict]:
    """Merge the sampled scores into time ranges worth analysing."""
    config = config or PlayabilityConfig()
    fps = scan["fps"]
    samples = scan["samples"]
    if not samples:
        return []

    spans: list[list[int]] = []
    inside = False
    for frame, share in samples:
        if not inside and share >= config.open_grass_share:
            spans.append([frame, frame])
            inside = True
        elif inside and share >= config.min_grass_share:
            spans[-1][1] = frame
        elif inside:
            inside = False

    pad = int(round(config.pad_s * fps))
    merge_gap = int(round(config.merge_gap_s * fps))
    padded: list[list[int]] = []
    for start, end in spans:
        start, end = max(0, start - pad), min(scan["frames"] - 1, end + pad)
        if padded and start - padded[-1][1] <= merge_gap:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])

    minimum = config.min_segment_s * fps
    return [
        {
            "start_frame": start,
            "end_frame": end,
            "start_s": round(start / fps, 2),
            "end_s": round(end / fps, 2),
            "duration_s": round((end - start + 1) / fps, 2),
        }
        for start, end in padded
        if end - start + 1 >= minimum
    ]


def screen(path: str, config: PlayabilityConfig | None = None) -> dict[str, Any]:
    """Scan a file and report what is worth sending to the GPU."""
    config = config or PlayabilityConfig()
    scan = scan_video(path, config)
    segments = playable_segments(scan, config)
    total = scan["frames"] / scan["fps"] if scan["fps"] else 0.0
    playable = sum(segment["duration_s"] for segment in segments)
    shares = [share for _, share in scan["samples"]]
    return {
        "video": path,
        "fps": scan["fps"],
        "frames": scan["frames"],
        "duration_s": round(total, 2),
        "sampled": len(scan["samples"]),
        "stride_frames": config.stride_frames,
        "median_grass_share": round(float(np.median(shares)), 3) if shares else 0.0,
        "segments": segments,
        "playable_s": round(playable, 2),
        "playable_share": round(playable / total, 4) if total else 0.0,
        "skipped_s": round(total - playable, 2),
    }


__all__ = ["PlayabilityConfig", "grass_share", "playable_segments", "scan_video", "screen"]
