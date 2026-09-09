"""Pure-numpy drop-in for `cython_bbox`.

Shipped with the engine rather than installed, because the original has no
Windows wheel and needs a compiler plus numpy headers on Linux — neither is
worth carrying for one function.  Same semantics as the Cython implementation,
including its inclusive +1 pixel convention, so the tracker behaves identically.
"""
import numpy as np

__all__ = ["bbox_overlaps"]


def bbox_overlaps(boxes, query_boxes):
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query = np.ascontiguousarray(query_boxes, dtype=np.float64)
    n, k = boxes.shape[0], query.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    iw = (np.minimum(boxes[:, None, 2], query[None, :, 2])
          - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1)
    ih = (np.minimum(boxes[:, None, 3], query[None, :, 3])
          - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1)
    np.clip(iw, 0, None, out=iw)
    np.clip(ih, 0, None, out=ih)

    inter = iw * ih
    area_b = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1))[:, None]
    area_q = ((query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1))[None, :]
    union = area_b + area_q - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)
