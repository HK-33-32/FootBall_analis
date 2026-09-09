"""Loading models once, and preparing crops on more than one core.

The stitching stage was measured at 432 s for 7.6 s of football with the GPU at
10% utilisation, so most of that time was not GPU work. Two causes, neither of
them algorithmic:

*The same network was loaded repeatedly.* ``gen_tracklets`` and
``tracklet_attributes`` run in one process and each built its own OSNet -- ten
seconds of weight loading, and a second copy in VRAM. They even reached the
same file through two different import paths (``torchreid`` and
``reid.torchreid``), which Python treats as two modules, so nothing was shared.
``shared_reid`` gives the process one extractor per weights file.

*Crops were prepared one at a time.* Decoding a frame, cropping a player and
resizing him for the network is CPU work that releases the GIL, so it spreads
across cores; done serially it left fifteen of sixteen idle while the GPU
waited. ``thread_map`` runs it on a pool sized to the machine.

Neither changes what the networks are shown, only when and on which core it is
prepared.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

_REID_CACHE: dict = {}
_POOL: ThreadPoolExecutor | None = None
# Beyond this the pool costs more in contention than it returns: the work is
# already releasing the GIL and the GPU is the next thing to wait for.
MAX_WORKERS = 8
# Below this a pool is pure overhead. It is low because the items here are not
# small: a CLIP preprocess is about 36 ms, against tens of microseconds to hand
# it to a thread, and the jersey head is called with only the handful of crops
# one frame holds.
MIN_ITEMS = 3


def workers() -> int:
    return max(1, min(MAX_WORKERS, os.cpu_count() or 4))


def pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=workers(), thread_name_prefix="prep")
    return _POOL


def thread_map(fn, items) -> list:
    """Map over items in order, on the pool when there is enough to be worth it."""
    items = list(items)
    if len(items) < MIN_ITEMS:
        return [fn(item) for item in items]
    return list(pool().map(fn, items))


def shared_reid(model_name: str, model_path, device):
    """One FeatureExtractor per weights file per process.

    Imported through a single canonical path so that callers reaching for
    ``torchreid`` and callers reaching for ``reid.torchreid`` end up on the same
    object instead of loading the file twice.
    """
    key = (model_name, str(model_path), str(device))
    if key not in _REID_CACHE:
        from torchreid.utils import FeatureExtractor

        _REID_CACHE[key] = FeatureExtractor(
            model_name=model_name, model_path=str(model_path), device=str(device)
        )
    return _REID_CACHE[key]


__all__ = ["shared_reid", "thread_map", "pool", "workers"]
