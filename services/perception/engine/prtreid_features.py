"""PRTReID embeddings for team clustering.

OSNet, the ReID network the pipeline ships with, is trained to tell *people*
apart — two players of the same team are exactly what it is built to separate.
That makes it a poor signal for "which team is this", which is why clustering on
its embeddings scored a silhouette of 0.16 while a plain torso colour histogram
scored 0.48.

PRTReID (Somers et al., the ReID model behind the SoccerNet game-state baseline)
is trained on football with a team-aware objective, so its embedding is supposed
to carry kit identity rather than personal identity.  This module wraps it
behind the same call shape as the OSNet extractor: crops in, one vector each.

Weights: https://zenodo.org/records/10653453 (prtreid-soccernet-baseline).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEFAULT_WEIGHTS = os.path.join("checkpoints", "prtreid", "prtreid-soccernet-baseline.pth.tar")
DEFAULT_BACKBONE = os.path.join("checkpoints", "prtreid")

# index order the SoccerNet baseline trained with
ROLES = ["ball", "goalkeeper", "other", "player", "referee"]


def _build_config(weights: str, backbone_dir: str):
    from prtreid.scripts.default_config import get_default_config

    cfg = get_default_config()
    cfg.project.logger.use_wandb = False
    cfg.project.logger.use_tensorboard = False
    cfg.data.height, cfg.data.width = 256, 128
    cfg.data.save_dir = ""
    cfg.model.name = "bpbreid"
    cfg.model.pretrained = True
    cfg.model.load_weights = weights
    cfg.model.bpbreid.backbone = "hrnet32"
    cfg.model.bpbreid.hrnet_pretrained_path = backbone_dir
    cfg.model.bpbreid.pooling = "gwap"
    cfg.model.bpbreid.normalization = "identity"
    cfg.model.bpbreid.last_stride = 1
    cfg.model.bpbreid.dim_reduce = "after_pooling"
    cfg.model.bpbreid.dim_reduce_output = 256
    cfg.model.bpbreid.learnable_attention_enabled = False
    cfg.model.bpbreid.mask_filtering_training = False
    cfg.model.bpbreid.mask_filtering_testing = False
    cfg.model.bpbreid.test_embeddings = ["globl"]
    cfg.model.bpbreid.test_use_target_segmentation = "none"
    cfg.model.bpbreid.shared_parts_id_classifier = False
    cfg.model.bpbreid.masks.type = "disk"
    cfg.model.bpbreid.masks.preprocess = "id"
    cfg.loss.name = "part_based"
    return cfg


class PRTReIDFeatures:
    """Global appearance embedding (and role guess) for a list of BGR crops."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS,
                 backbone_dir: str = DEFAULT_BACKBONE, device: str = "cuda"):
        from prtreid.tools.feature_extractor import FeatureExtractor

        self.device = device
        cfg = _build_config(weights, backbone_dir)
        self.extractor = FeatureExtractor(cfg, model_path=weights, device=device,
                                          image_size=(cfg.data.height, cfg.data.width),
                                          verbose=False)
        self.test_embeddings = cfg.model.bpbreid.test_embeddings

    @staticmethod
    def available(weights: str = DEFAULT_WEIGHTS) -> bool:
        return os.path.isfile(weights)

    @torch.no_grad()
    def __call__(self, crops, chunk: int = 64):
        """crops: list of BGR arrays -> (embeddings [N, D], roles [N])."""
        from prtreid.utils.tools import extract_test_embeddings

        vectors, roles = [], []
        for start in range(0, len(crops), chunk):
            part = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops[start:start + chunk]]
            result = self.extractor(part)
            embeddings, _, _, _, role_scores = extract_test_embeddings(
                result, self.test_embeddings)
            vec = embeddings.float().cpu().numpy()
            if vec.ndim == 3:          # (N, parts, dim) with a single part
                vec = vec[:, 0, :]
            vectors.append(vec)
            if role_scores is not None and "globl" in role_scores:
                idx = role_scores["globl"].argmax(dim=1).cpu().numpy()
                roles.extend(ROLES[i] if i < len(ROLES) else "other" for i in idx)
            else:
                roles.extend(["player"] * len(part))
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32), []
        return np.concatenate(vectors, axis=0), roles
