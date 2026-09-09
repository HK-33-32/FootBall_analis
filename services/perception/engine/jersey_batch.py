"""Batched inference for the CLIP role / jersey-number / colour head.

Same maths as ``predict_role_and_jersey_batch_clip`` in
``inference_soccernetGSR.py``, but it runs one forward per batch instead of one
per crop and encodes the colour prompts once instead of once per frame.
Verified to return identical predictions.
"""
from __future__ import annotations

import clip
import torch
import torch.nn.functional as F

from fast_prep import thread_map

ROLE_MAPPING = {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball", 4: "Other"}
COMMON_COLORS = ["Red", "Blue", "Green", "Yellow", "White", "Black", "Orange",
                 "Purple", "Pink", "Brown", "Grey", "Charcoal", "Neon yellow"]


def get_number(digit1, digit2):
    """10 means 'no digit / not visible'; 100 means 'unknown number'."""
    if digit1 == 10:
        return 100 if digit2 == 10 else digit2
    return digit1 * 10 + digit2 if digit2 != 10 else digit1


class BatchedJerseyClassifier:
    def __init__(self, jersey_model, device, chunk=96):
        self.model = jersey_model
        self.device = device
        self.chunk = chunk
        with torch.no_grad():
            feats = jersey_model.clip_model.encode_text(
                clip.tokenize(COMMON_COLORS).to(device))
        self.color_features = F.normalize(feats, p=2, dim=1).float()

    @torch.no_grad()
    def predict(self, pil_images):
        """pil_images: list of PIL.Image -> list of (role, number, colour)."""
        results = []
        for start in range(0, len(pil_images), self.chunk):
            part = pil_images[start:start + self.chunk]
            # CLIP's preprocess is a bicubic resize and a normalise per crop:
            # CPU work that releases the GIL, and the single largest item in the
            # tracking stage at 47 s of its 137 before it was spread out
            batch = torch.stack(thread_map(self.model.preprocess, part))
            batch = batch.to(self.device, non_blocking=True)
            out = self.model(batch)

            roles = out["role_logits"].argmax(1).tolist()
            digit1 = out["digit1_logits"].argmax(1).tolist()
            digit2 = out["digit2_logits"].argmax(1).tolist()
            colors = (F.normalize(out["color_embedding"], p=2, dim=1).float()
                      @ self.color_features.T).argmax(1).tolist()

            for role, d1, d2, color in zip(roles, digit1, digit2, colors):
                results.append((ROLE_MAPPING[role], get_number(d1, d2),
                                COMMON_COLORS[color]))
        return results
