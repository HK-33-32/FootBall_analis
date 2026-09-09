"""Versioned, order-independent circle sampling for the legacy calibration adapter."""

import numpy as np

SAMPLER_VERSION = "circle-pcg64-v1"


def sample_circle_indices(population: int, count: int, *, seed: int | None = None) -> np.ndarray:
    """An explicit seed isolates sampling from worker scheduling and other frames.

    None preserves the legacy global RNG exactly. No seed is selected using
    validation labels; seeded runs must record the seed and sampler version.
    """
    if population < 0 or count < 0:
        raise ValueError("population and count must be nonnegative")
    size = min(population, count)
    if seed is None:
        return np.random.choice(population, size=size, replace=False)
    if seed < 0:
        raise ValueError("calibration seed must be nonnegative")
    return np.random.Generator(np.random.PCG64(seed)).choice(population, size=size, replace=False)
