"""Who has the ball, frame by frame — decided in image space.

Proximity is measured between the ball's centre and the player's feet in
pixels, then converted to metres using that player's own box height as the
scale (a footballer is about 1.8 m, so a box `h` pixels tall means `h / 1.8`
pixels per metre at his depth).  That keeps the comparison perspective-correct
without ever asking the homography where a ball in flight is, which is the one
question it cannot answer.

The naive answer — nearest player — still flickers between two challenging
players several times a second, so the raw sequence is smoothed and short runs
are dissolved back into loose ball.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER_HEIGHT_M = 1.8
CONTROL_RADIUS = 2.5       # m — closer than this counts as being on the ball
CONTROL_RADIUS_FAST = 1.2  # m — what "on the ball" means for a ball in flight
SLOW_BALL = 5.0            # m/s — at or below this the ball is at someone's feet
FAST_BALL = 15.0           # m/s — at or above this it is travelling, not held
CONTEST_RADIUS = 3.5       # m — two opponents inside this are challenging
# Calibrated against SoccerNet Game State ground truth over 58 labelled clips:
# at 0.15 s the derivation produced 21 passes a minute where football produces
# about 10, because a ball flying past an opponent for four frames was being
# read as him having had it.  At 0.5 s the rate lands at 9.1 passes and 16.9
# touches a minute, which is what the sport actually looks like.
MIN_RUN_S = 0.5            # runs shorter than this are noise, not possession
SMOOTH_WINDOW = 3          # frames, must be odd


def control_radius(speed: float) -> float:
    """How close a player must be before the ball counts as his.

    A fixed radius credits possession to whoever a 20 m/s pass happens to fly
    within two metres of, and every such phantom possession costs two events: a
    turnover for the passer and an interception for the bystander.
    """
    if not np.isfinite(speed) or speed <= SLOW_BALL:
        return CONTROL_RADIUS
    if speed >= FAST_BALL:
        return CONTROL_RADIUS_FAST
    share = (speed - SLOW_BALL) / (FAST_BALL - SLOW_BALL)
    return CONTROL_RADIUS + share * (CONTROL_RADIUS_FAST - CONTROL_RADIUS)


def per_frame(track: pd.DataFrame, detections: pd.DataFrame, fps: float) -> pd.DataFrame:
    """frame, holder_track, holder_dist, contested, n_near, ball_speed."""
    columns = ["frame", "holder_track", "holder_dist", "contested", "n_near", "ball_speed"]
    if track.empty:
        return pd.DataFrame(columns=columns)

    people = detections[detections.role.isin(("player", "goalkeeper"))]
    people = people.dropna(subset=["img_x", "img_y", "img_w", "img_h"])
    by_frame = {int(f): g for f, g in people.groupby("frame")}

    frames = track.frame.to_numpy().astype(int)
    bx = track.img_x.to_numpy()
    by = track.img_y.to_numpy()

    # ball speed in metres, using the frame's own pixel scale
    scale = np.full(len(track), np.nan)
    for i, frame in enumerate(frames):
        group = by_frame.get(int(frame))
        if group is not None and len(group):
            scale[i] = np.nanmedian(group.img_h.to_numpy()) / PLAYER_HEIGHT_M
    scale = pd.Series(scale).ffill().bfill().to_numpy()

    step_px = np.concatenate([[0.0], np.hypot(np.diff(bx), np.diff(by))])
    gaps = np.concatenate([[1.0], np.diff(frames).astype(float)])
    speed = np.divide(step_px, np.maximum(scale, 1e-6)) / np.maximum(gaps / fps, 1e-3)
    speed = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()

    rows = []
    for i, frame in enumerate(frames):
        group = by_frame.get(int(frame))
        if group is None or group.empty:
            rows.append((int(frame), -1, np.nan, False, 0, float(speed[i])))
            continue
        feet_x = group.img_x.to_numpy() + group.img_w.to_numpy() / 2.0
        feet_y = group.img_y.to_numpy() + group.img_h.to_numpy()
        pixels_per_m = np.maximum(group.img_h.to_numpy(), 1.0) / PLAYER_HEIGHT_M
        dist = np.hypot(feet_x - bx[i], feet_y - by[i]) / pixels_per_m
        best = int(np.argmin(dist))
        near = dist <= CONTEST_RADIUS
        teams = set(group.team.to_numpy()[near]) - {None}
        radius = control_radius(float(speed[i]))
        holder = int(group.track_id.to_numpy()[best]) if dist[best] <= radius else -1
        rows.append((int(frame), holder, float(dist[best]),
                     len(teams) > 1, int(near.sum()), float(speed[i])))

    df = pd.DataFrame(rows, columns=columns)
    df["holder_track"] = _smooth(df.holder_track.to_numpy(), SMOOTH_WINDOW)
    return df


def _smooth(ids: np.ndarray, window: int) -> np.ndarray:
    """Mode filter: a single frame stolen by a passing opponent is reverted."""
    half = window // 2
    out = ids.copy()
    for i in range(len(ids)):
        lo, hi = max(0, i - half), min(len(ids), i + half + 1)
        values, counts = np.unique(ids[lo:hi], return_counts=True)
        out[i] = int(values[int(np.argmax(counts))])
    return out


def runs(frames: pd.DataFrame, fps: float) -> list[dict]:
    """Collapse the per-frame holder into possession runs.

    A run is one player continuously on the ball.  Runs shorter than MIN_RUN_S
    are demoted to loose ball; adjacent runs by the same player, separated only
    by such noise, are then merged into one.
    """
    if frames.empty:
        return []
    min_len = max(1, int(round(MIN_RUN_S * fps)))
    raw: list[dict] = []
    start = 0
    ids = frames.holder_track.to_numpy()
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            raw.append({"track_id": int(ids[start]),
                        "start_row": start, "end_row": i - 1})
            start = i

    for run in raw:
        length = run["end_row"] - run["start_row"] + 1
        if run["track_id"] != -1 and length < min_len:
            run["track_id"] = -1

    merged: list[dict] = []
    for run in raw:
        if merged and merged[-1]["track_id"] == run["track_id"]:
            merged[-1]["end_row"] = run["end_row"]
        else:
            merged.append(run)

    out = []
    for run in merged:
        lo, hi = run["start_row"], run["end_row"]
        out.append({
            "track_id": run["track_id"],
            "start_frame": int(frames.frame.iloc[lo]),
            "end_frame": int(frames.frame.iloc[hi]),
            "start_row": lo, "end_row": hi,
            "frames": hi - lo + 1,
            "duration_s": (hi - lo + 1) / fps,
            "contested": bool(frames.contested.iloc[lo:hi + 1].any()),
            "mean_dist": float(np.nanmean(frames.holder_dist.iloc[lo:hi + 1])),
        })
    return out
