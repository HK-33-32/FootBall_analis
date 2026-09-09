"""Broadcast-style overlay: ground rings and floating name plates, not boxes.

A rectangle is what a detector outputs; it is not what a viewer wants to look
at.  This draws what a football game does — a team-coloured ring on the grass
under each player, a name plate floating above the head, a chevron over whoever
is on the ball, and a clean vector minimap built from the pitch coordinates the
calibration already gives us.

Drop-in replacement for `visualize_prediction_results.visualize_predictions`,
so the runner can switch between the two by config alone.
"""
import json
import os

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX

# BGR.  Kept aligned with the box renderer so a run does not change sides when
# the style is switched: left is warm, right is cool.
TEAM_COLORS = {
    "left": (70, 90, 245),      # red
    "right": (245, 170, 70),    # blue
}
GK_COLOR = (235, 80, 235)       # magenta: the keeper's kit matches neither team
REF_COLOR = (60, 230, 245)      # amber
BALL_COLOR = (255, 255, 255)
NEUTRAL = (200, 200, 200)

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def format_image_name(image_id):
    return "%06d.jpg" % int(str(image_id)[-6:])


# --- drawing primitives -----------------------------------------------------
def _rounded_rect(img, p1, p2, radius, color, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy, start in ((x1 + radius, y1 + radius, 180), (x2 - radius, y1 + radius, 270),
                          (x2 - radius, y2 - radius, 0), (x1 + radius, y2 - radius, 90)):
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, start + 90,
                    color, thickness)


def _shade(color, factor):
    return tuple(int(max(0, min(255, c * factor))) for c in color)


def _ground_ring(overlay, image, feet, width, color):
    """The marker a football game puts under a player, in ground perspective."""
    rx = max(6, int(width * 0.55))
    ry = max(3, int(width * 0.21))
    cv2.ellipse(overlay, feet, (rx, ry), 0, 0, 360, _shade(color, 0.55), -1)
    cv2.ellipse(image, feet, (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.ellipse(image, feet, (max(2, rx - 4), max(1, ry - 2)), 0, 0, 360,
                _shade(color, 1.35), 1, cv2.LINE_AA)


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _name_plate(image, overlay, anchor, label, number, color, scale, placed):
    """Floating tag above the head: number badge on the left, name on the right.

    Players bunch up, and two tags on top of each other are worse than none, so
    a tag that would collide is lifted until it clears the ones already drawn.
    """
    pad = max(4, int(8 * scale))
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
    badge_w = 0
    if number:
        (nw, _), _ = cv2.getTextSize(number, FONT, scale, 1)
        badge_w = nw + 2 * pad
    height = th + 2 * pad
    width = badge_w + tw + 2 * pad
    x = int(anchor[0] - width / 2)
    y = int(anchor[1] - height)

    h, w = image.shape[:2]
    x = max(2, min(x, w - width - 2))
    y = max(2, min(y, h - height - 2))
    for _ in range(8):
        rect = (x, y, x + width, y + height)
        clash = next((r for r in placed if _overlaps(rect, r)), None)
        if clash is None or y - (height + 3) < 2:
            break
        y -= height + 3
    placed.append((x, y, x + width, y + height))

    _rounded_rect(overlay, (x, y), (x + width, y + height), height // 3,
                  _shade(color, 0.35), -1)
    if badge_w:
        _rounded_rect(overlay, (x, y), (x + badge_w, y + height), height // 3,
                      color, -1)
    _rounded_rect(image, (x, y), (x + width, y + height), height // 3,
                  _shade(color, 1.25), 1)
    if number:
        cv2.putText(image, number, (x + pad, y + height - pad), FONT, scale,
                    (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(image, label, (x + badge_w + pad, y + height - pad), FONT, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return x + width // 2, y


def _chevron(image, apex, size, color):
    """The marker over the player currently in control."""
    pts = np.array([[apex[0], apex[1]],
                    [apex[0] - size, apex[1] - int(size * 1.3)],
                    [apex[0] + size, apex[1] - int(size * 1.3)]], np.int32)
    cv2.fillPoly(image, [pts], color, cv2.LINE_AA)
    cv2.polylines(image, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def _ball_marker(image, overlay, centre, radius):
    for i, alpha in enumerate((0.9, 0.55, 0.3)):
        cv2.circle(overlay, centre, int(radius * (1.6 + 0.9 * i)),
                   _shade(BALL_COLOR, alpha), 2, cv2.LINE_AA)
    cv2.circle(image, centre, max(2, int(radius * 0.7)), BALL_COLOR, -1, cv2.LINE_AA)


# --- minimap ----------------------------------------------------------------
def _pitch_panel(width, height):
    """A vector pitch, drawn once and reused for every frame."""
    panel = np.full((height, width, 3), (35, 75, 35), np.uint8)
    line = (225, 235, 225)
    m = int(width * 0.04)
    x0, y0, x1, y1 = m, m, width - m, height - m
    cv2.rectangle(panel, (x0, y0), (x1, y1), line, 1, cv2.LINE_AA)
    cv2.line(panel, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), line, 1, cv2.LINE_AA)
    cv2.circle(panel, ((x0 + x1) // 2, (y0 + y1) // 2),
               int((x1 - x0) * 9.15 / PITCH_LENGTH), line, 1, cv2.LINE_AA)
    box_w = int((x1 - x0) * 16.5 / PITCH_LENGTH)
    box_h = int((y1 - y0) * 40.32 / PITCH_WIDTH)
    cy = (y0 + y1) // 2
    cv2.rectangle(panel, (x0, cy - box_h // 2), (x0 + box_w, cy + box_h // 2),
                  line, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (x1 - box_w, cy - box_h // 2), (x1, cy + box_h // 2),
                  line, 1, cv2.LINE_AA)
    return panel, (x0, y0, x1, y1)


_PANEL_CACHE = {}


def _minimap(size, entries):
    panel, bounds = _PANEL_CACHE.setdefault(size, _pitch_panel(*size))
    canvas = panel.copy()
    x0, y0, x1, y1 = bounds
    for x_m, y_m, color, is_ball in entries:
        px = int(x0 + (x_m + PITCH_LENGTH / 2) / PITCH_LENGTH * (x1 - x0))
        py = int(y0 + (y_m + PITCH_WIDTH / 2) / PITCH_WIDTH * (y1 - y0))
        if not (0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]):
            continue
        if is_ball:
            cv2.circle(canvas, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 4, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


# --- per-detection styling --------------------------------------------------
def _style(prediction):
    attributes = prediction.get("attributes") or {}
    role = str(attributes.get("role") or "").lower()
    team = str(attributes.get("team") or "").lower()
    jersey = attributes.get("jersey")
    jersey = None if jersey in (None, "", "100", 100) else str(jersey)
    name = attributes.get("name")

    if role == "ball":
        return "ball", BALL_COLOR, None, None
    if role in ("referee", "other"):
        return role, REF_COLOR, "REF", None
    if role == "goalkeeper":
        side = {"left": "L", "right": "R"}.get(team, "?")
        return role, GK_COLOR, name or ("GK " + side), jersey
    color = TEAM_COLORS.get(team, NEUTRAL)
    label = name or ("#" + jersey if jersey else "ID %s" % prediction.get("track_id"))
    if name and jersey:
        return role, color, name, jersey
    return role, color, label, None


def visualize_predictions(image_folder, output_folder, json_file_path):
    base_result_dir = os.path.join(output_folder, "Predict_Visualization")
    os.makedirs(base_result_dir, exist_ok=True)

    if not os.path.isfile(json_file_path):
        print("Error: JSON file not found at %s" % json_file_path)
        return
    with open(json_file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    by_image, video_of = {}, {}
    for prediction in data.get("predictions", []):
        image_id = prediction.get("image_id")
        by_image.setdefault(image_id, []).append(prediction)
        video_of[image_id] = prediction.get("video_id")

    for image_id, predictions in by_image.items():
        file_name = format_image_name(image_id)
        result_dir = os.path.join(base_result_dir, "SNGS-%s" % video_of.get(image_id, "unknown"))
        os.makedirs(result_dir, exist_ok=True)

        image = cv2.imread(os.path.join(image_folder, file_name))
        if image is None:
            print("Error: Failed to load image at %s" % file_name)
            continue
        overlay = image.copy()
        height, width = image.shape[:2]

        ball_feet = None
        for prediction in predictions:
            box = prediction.get("bbox_image")
            if box and str((prediction.get("attributes") or {}).get("role", "")).lower() == "ball":
                ball_feet = (int(box["x"] + box["w"] / 2), int(box["y"] + box["h"] / 2))

        drawn, radar = [], []
        for prediction in predictions:
            role, color, label, number = _style(prediction)
            box = prediction.get("bbox_image")
            pitch = prediction.get("bbox_pitch")
            if pitch and pitch.get("x_bottom_middle") is not None:
                radar.append((pitch["x_bottom_middle"], pitch["y_bottom_middle"],
                              color, role == "ball"))
            if not box:
                continue
            x, y = int(box["x"]), int(box["y"])
            w, h = int(box["w"]), int(box["h"])
            if role == "ball":
                _ball_marker(image, overlay, (x + w // 2, y + h // 2), max(4, w // 2))
                continue
            feet = (x + w // 2, y + h)
            _ground_ring(overlay, image, feet, w, color)
            drawn.append((prediction, feet, (x, y, w, h), color, label, number))

        # Name plates last, so no ring is drawn over a label.
        carrier = None
        if ball_feet and drawn:
            carrier = min(drawn, key=lambda d: (d[1][0] - ball_feet[0]) ** 2 +
                          (d[1][1] - ball_feet[1]) ** 2)
            if (carrier[1][0] - ball_feet[0]) ** 2 + (carrier[1][1] - ball_feet[1]) ** 2 > \
                    (3.5 * max(carrier[2][2], 20)) ** 2:
                carrier = None

        placed = []
        # Nearest players first: the ones in front keep their natural position
        # and the ones behind are the ones that get lifted.
        for entry in sorted(drawn, key=lambda d: -d[1][1]):
            _, feet, (x, y, w, h), color, label, number = entry
            scale = float(np.clip(w / 130.0, 0.38, 0.8))
            top = _name_plate(image, overlay, (x + w // 2, y - int(8 * scale)),
                              label, number, color, scale, placed)
            if carrier is entry:
                _chevron(image, (top[0], top[1] - 4), max(5, int(10 * scale)), color)

        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        panel_w = max(200, width // 5)
        panel = _minimap((panel_w, int(panel_w * PITCH_WIDTH / PITCH_LENGTH)), radar)
        ph, pw = panel.shape[:2]
        ox, oy = width - pw - 24, 24
        region = image[oy:oy + ph, ox:ox + pw]
        cv2.addWeighted(panel, 0.94, region, 0.06, 0, region)
        cv2.rectangle(image, (ox - 1, oy - 1), (ox + pw + 1, oy + ph + 1),
                      (15, 15, 15), 3, cv2.LINE_AA)
        cv2.rectangle(image, (ox, oy), (ox + pw, oy + ph), (235, 235, 235), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(result_dir, "%s.jpg" % os.path.splitext(file_name)[0]),
                    image)
