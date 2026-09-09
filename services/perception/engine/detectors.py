"""Pluggable person/ball detectors for the GSR pipeline.

Every detector returns, per frame, an ``(N, 5)`` array of
``[x1, y1, x2, y2, score]`` in *original image* coordinates, which is what
``Deep_EIoU.update`` expects.
"""
from __future__ import annotations

import numpy as np
import torch

# COCO ids used by the RF-DETR checkpoints
COCO_PERSON = 1
COCO_SPORTS_BALL = 37


class YoloxDetector:
    """The SoccerNet-finetuned YOLOX-x that ships with the repository."""

    name = "YOLOX-x (SoccerNet)"

    def __init__(self, cfg, device, fp16=True):
        from yolox.exp import get_exp
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        self._preproc = preproc
        self._postprocess = postprocess

        exp = get_exp(cfg["EXP_FILE"], "yolo-x")
        exp.test_conf = cfg.get("CONF", 0.1)
        exp.nmsthre = cfg.get("NMS_THRESH", 0.7)
        exp.test_size = (800, 1440)
        self.exp = exp

        model = exp.get_model().to(device).eval()
        model.load_state_dict(torch.load(cfg["MODEL_PATH"], weights_only=False)["model"])
        self.fp16 = bool(fp16)
        self.model = model.half() if self.fp16 else model
        self.device = device
        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)

    @torch.no_grad()
    def detect_batch(self, frames):
        procs, ratios = [], []
        for frame in frames:
            proc, ratio = self._preproc(frame, self.exp.test_size, self.means, self.stds)
            procs.append(proc)
            ratios.append(ratio)
        batch = torch.from_numpy(np.stack(procs)).to(self.device)
        batch = batch.half() if self.fp16 else batch.float()
        outputs = self._postprocess(self.model(batch), self.exp.num_classes,
                                    self.exp.test_conf, self.exp.nmsthre)

        results = []
        for out, ratio in zip(outputs, ratios):
            if out is None or not len(out):
                results.append(np.zeros((0, 5), dtype=np.float32))
                continue
            out = out.float().cpu().numpy()
            boxes = out[:, :4] / ratio
            scores = out[:, 4] * out[:, 5]
            results.append(np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32))
        return results


class RFDetrDetector:
    """RF-DETR (Roboflow real-time DETR), COCO weights, NMS-free.

    Keeps the ``person`` and ``sports ball`` classes; on broadcast football
    footage that is exactly the set the tracker needs.
    """

    _SIZES = {
        "nano": ("RFDETRNano", 640),
        "small": ("RFDETRSmall", 640),
        "medium": ("RFDETRMedium", 1024),
        "base": ("RFDETRBase", 1008),
        "large": ("RFDETRLarge", 1120),
    }

    def __init__(self, cfg, device, batch_size=4):
        import rfdetr

        size = str(cfg.get("RFDETR_MODEL", "medium")).lower()
        if size not in self._SIZES:
            raise ValueError("unknown RF-DETR size %r" % size)
        cls_name, default_res = self._SIZES[size]
        resolution = int(cfg.get("RFDETR_RESOLUTION") or default_res)

        self.name = "RF-DETR %s @%d" % (size, resolution)
        self.threshold = float(cfg.get("RFDETR_CONF", 0.35))
        self.ball_threshold = float(cfg.get("RFDETR_BALL_CONF", 0.05))

        # The ball is the one object whose detector score cannot be compared
        # with a person's: at 1080p it is about 16 px across, and the model
        # shrinks the frame to `resolution` before it ever sees it.  Its median
        # score lands near 0.55, under the tracker's NEW_TRACK_THRESH of 0.7,
        # so it was being detected and then never given a track.  The three
        # settings below deal with that: keep low-scoring candidates, pick one
        # per frame by physics rather than by score, and hand the survivor to
        # the tracker at a score a person would have earned.
        self.ball_refine = bool(cfg.get("BALL_REFINE", True))
        self.ball_crop = int(cfg.get("BALL_CROP", 512))
        self.ball_track_score = float(cfg.get("BALL_TRACK_SCORE", 0.85))
        # a struck ball crosses about 1600 px of a 1080p frame in a second;
        # expressed per second the gate stays correct when fps changes
        fps = float(cfg.get("FPS", 25) or 25)
        self.ball_max_step = float(cfg.get("BALL_MAX_STEP_PX_PER_S", 1600.0)) / fps
        self.ball_reacquire = int(cfg.get("BALL_REACQUIRE_FRAMES", 8))
        self._frame_index = 0
        self._last_ball = None          # (frame_index, cx, cy)
        self.ball_stats = {"frames": 0, "accepted": 0, "from_crop": 0}

        model = getattr(rfdetr, cls_name)(resolution=resolution)
        model.optimize_for_inference(batch_size=batch_size, dtype=torch.float16)
        self.model = model
        self.batch_size = batch_size
        # torch.compile builds the graph on the first forward: do it now so the
        # first real batch is not 3 s slower than the rest
        warm = np.zeros((batch_size, resolution, resolution, 3), dtype=np.uint8)
        self.model.predict(list(warm), threshold=0.9)

    def detect_batch(self, frames):
        # rfdetr expects RGB; the pipeline reads frames with cv2 (BGR)
        rgb = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames]
        # the model is compiled for a fixed batch size: pad the tail batch
        real = len(rgb)
        if real < self.batch_size:
            rgb = rgb + [rgb[-1]] * (self.batch_size - real)
        preds = self.model.predict(rgb, threshold=min(self.threshold, self.ball_threshold))
        if not isinstance(preds, list):
            preds = [preds]
        preds = preds[:real]

        people, ball_candidates = [], []
        for det in preds:
            people.append(self._rows(det, COCO_PERSON, self.threshold))
            ball_candidates.append(self._rows(det, COCO_SPORTS_BALL, self.ball_threshold))

        balls = self._pick_balls(frames, ball_candidates)
        results = []
        for person_rows, ball in zip(people, balls):
            parts = [person_rows]
            if ball is not None:
                parts.append(ball[None, :])
            results.append(np.concatenate(parts, axis=0).astype(np.float32)
                           if any(len(p) for p in parts)
                           else np.zeros((0, 5), dtype=np.float32))
        return results

    @staticmethod
    def _rows(det, class_id, floor):
        if det.xyxy is None or not len(det.xyxy):
            return np.zeros((0, 5), dtype=np.float32)
        cls = np.asarray(det.class_id)
        conf = np.asarray(det.confidence, dtype=np.float32)
        keep = (cls == class_id) & (conf >= floor)
        if not keep.any():
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate([np.asarray(det.xyxy, dtype=np.float32)[keep],
                               conf[keep, None]], axis=1).astype(np.float32)

    def _pick_balls(self, frames, candidates):
        """At most one ball per frame, chosen by where a ball could have gone.

        Dropping the score floor turns up the ball far more often, but it also
        turns up seven false candidates a frame.  Selecting by continuity from
        the last accepted position rather than by score keeps the extra recall
        and throws the noise away; frames left empty are then retried on a crop
        that follows the ball, where it is no longer a 16-pixel speck.
        """
        chosen = [None] * len(frames)
        for i, rows in enumerate(candidates):
            index = self._frame_index + i
            self.ball_stats["frames"] += 1
            pick = self._gate(rows, index)
            if pick is not None:
                chosen[i] = pick
                self._remember(pick, index)

        if self.ball_refine:
            self._refine(frames, chosen)

        self._frame_index += len(frames)
        for pick in chosen:
            if pick is not None:
                self.ball_stats["accepted"] += 1
        return [None if p is None else self._as_track_row(p) for p in chosen]

    def _gate(self, rows, index):
        if not len(rows):
            return None
        centres = np.stack([(rows[:, 0] + rows[:, 2]) / 2,
                            (rows[:, 1] + rows[:, 3]) / 2], axis=1)
        if self._last_ball is None:
            return rows[int(np.argmax(rows[:, 4]))]
        last_index, lx, ly = self._last_ball
        gap = max(1, index - last_index)
        if gap > self.ball_reacquire:
            # the ball has been gone long enough that continuity says nothing;
            # start again from the most confident candidate
            return rows[int(np.argmax(rows[:, 4]))]
        reachable = np.where(np.hypot(centres[:, 0] - lx, centres[:, 1] - ly)
                             <= self.ball_max_step * gap)[0]
        if not len(reachable):
            return None
        return rows[int(reachable[np.argmax(rows[reachable, 4])])]

    def _remember(self, row, index):
        self._last_ball = (index, float((row[0] + row[2]) / 2),
                           float((row[1] + row[3]) / 2))

    def _refine(self, frames, chosen):
        """Second look at the frames that came up empty, on a following crop."""
        if self._last_ball is None:
            return
        height, width = frames[0].shape[:2]
        crop = min(self.ball_crop, height, width)
        pending, crops, offsets = [], [], []
        for i, pick in enumerate(chosen):
            if pick is not None:
                continue
            index = self._frame_index + i
            last_index, cx, cy = self._last_ball
            if index - last_index > self.ball_reacquire:
                continue
            x0 = int(np.clip(cx - crop / 2, 0, width - crop))
            y0 = int(np.clip(cy - crop / 2, 0, height - crop))
            pending.append(i)
            offsets.append((x0, y0))
            crops.append(np.ascontiguousarray(
                frames[i][y0:y0 + crop, x0:x0 + crop, ::-1]))
        if not crops:
            return

        for start in range(0, len(crops), self.batch_size):
            group = crops[start:start + self.batch_size]
            real = len(group)
            if real < self.batch_size:
                group = group + [group[-1]] * (self.batch_size - real)
            preds = self.model.predict(group, threshold=self.ball_threshold)
            if not isinstance(preds, list):
                preds = [preds]
            for k, det in enumerate(preds[:real]):
                rows = self._rows(det, COCO_SPORTS_BALL, self.ball_threshold)
                if not len(rows):
                    continue
                x0, y0 = offsets[start + k]
                rows[:, [0, 2]] += x0
                rows[:, [1, 3]] += y0
                i = pending[start + k]
                pick = self._gate(rows, self._frame_index + i)
                if pick is None:
                    continue
                chosen[i] = pick
                self.ball_stats["from_crop"] += 1
                self._remember(pick, self._frame_index + i)

    def _as_track_row(self, row):
        """Give the ball a score the tracker will start a track for.

        The tracker's thresholds were tuned on people.  Rescaling here, after
        the candidate has already survived the continuity gate, is what lets a
        ball hold a track id instead of being seen and immediately forgotten.
        """
        out = row.copy()
        out[4] = self.ball_track_score + (1.0 - self.ball_track_score) * float(row[4])
        return out


def build_detector(cfg, device, batch_size=4):
    kind = str(cfg.get("TYPE", "yolox")).lower()
    if kind == "rfdetr":
        return RFDetrDetector(cfg, device, batch_size=batch_size)
    if kind == "yolox":
        return YoloxDetector(cfg, device, fp16=cfg.get("FP16", True))
    raise ValueError("unknown detector %r" % kind)
