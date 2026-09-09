"""Build a continuous ball trajectory out of sparse per-frame detections.

The trajectory is kept in *image* coordinates, not pitch coordinates, and that
is the whole point of this module.  A pitch coordinate is produced by casting a
ray through the detection and intersecting it with the ground plane, which is
correct for a player — he stands on the ground — and wrong for a ball the
moment it leaves it.  Measured against SoccerNet ground truth, 42% of ball
annotations project outside the pitch entirely, some to 128 metres off the
touchline, and the median distance from the ball to the nearest player comes
out at 5.5 m — a number football does not produce.  In image space the same
ground truth gives a median of 1.2 m.

So the ball lives in pixels here.  Pitch coordinates are carried along for
whatever wants them but are never used for proximity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPEED_PX_PER_S = 1600.0   # a struck ball across a 1080p frame
GATE_SLACK_PX = 40.0
MAX_GAP_S = 1.0               # interpolate across gaps no longer than this


def build_track(ball: pd.DataFrame, fps: float) -> pd.DataFrame:
    """One row per frame: frame, img_x, img_y, x, y, observed."""
    columns = ["frame", "img_x", "img_y", "x", "y", "observed"]
    if ball.empty:
        return pd.DataFrame(columns=columns)

    candidates = ball.dropna(subset=["img_x", "img_y", "img_w", "img_h"]).copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates["cx"] = candidates.img_x + candidates.img_w / 2.0
    candidates["cy"] = candidates.img_y + candidates.img_h / 2.0

    max_step = MAX_SPEED_PX_PER_S / max(fps, 1e-3)
    accepted = []
    for frame, group in candidates.groupby("frame"):
        pts = group[["cx", "cy", "x", "y"]].to_numpy()
        if accepted:
            pf, px, py = accepted[-1][0], accepted[-1][1], accepted[-1][2]
            gap = max(1, int(frame) - int(pf))
            dist = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
            best = int(np.argmin(dist))
            if dist[best] > max_step * gap + GATE_SLACK_PX:
                continue
        else:
            best = 0
        accepted.append((int(frame), float(pts[best, 0]), float(pts[best, 1]),
                         float(pts[best, 2]), float(pts[best, 3])))

    if not accepted:
        return pd.DataFrame(columns=columns)

    frames = np.array([a[0] for a in accepted])
    cx = np.array([a[1] for a in accepted])
    cy = np.array([a[2] for a in accepted])
    px = np.array([a[3] for a in accepted])
    py = np.array([a[4] for a in accepted])

    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    rows = []
    for i, frame in enumerate(frames):
        rows.append((int(frame), cx[i], cy[i], px[i], py[i], True))
        if i + 1 < len(frames):
            gap = int(frames[i + 1] - frame)
            if 1 < gap <= max_gap:
                for step in range(1, gap):
                    t = step / gap
                    rows.append((int(frame + step),
                                 float(cx[i] + t * (cx[i + 1] - cx[i])),
                                 float(cy[i] + t * (cy[i + 1] - cy[i])),
                                 float(px[i] + t * (px[i + 1] - px[i])),
                                 float(py[i] + t * (py[i + 1] - py[i])),
                                 False))

    track = pd.DataFrame(rows, columns=columns).sort_values("frame")
    track = track.drop_duplicates("frame").reset_index(drop=True)
    if len(track) >= 3:
        track["img_x"] = track.img_x.rolling(3, center=True, min_periods=1).median()
        track["img_y"] = track.img_y.rolling(3, center=True, min_periods=1).median()
    return track


def coverage(track: pd.DataFrame, frames_total: int) -> dict:
    if track.empty or not frames_total:
        return {"frames": 0, "observed": 0, "interpolated": 0, "share": 0.0}
    observed = int(track.observed.sum())
    return {
        "frames": int(len(track)),
        "observed": observed,
        "interpolated": int(len(track) - observed),
        "share": round(len(track) / frames_total, 3),
    }
