import numpy as np
import pytest

from football_intelligence.calibration_sampling import sample_circle_indices


def test_unset_seed_is_exact_legacy_sampling():
    np.random.seed(12)
    expected = np.random.choice(500, size=100, replace=False)
    np.random.seed(12)
    assert np.array_equal(sample_circle_indices(500, 100), expected)


def test_seeded_sampling_is_independent_of_global_rng_and_call_order():
    expected = sample_circle_indices(500, 100, seed=0)
    for i in range(8):
        np.random.seed(i)
        sample_circle_indices(300, 80, seed=i)
        assert np.array_equal(sample_circle_indices(500, 100, seed=0), expected)
    assert len(set(expected)) == 100


def test_seeded_sampling_does_not_change_global_rng():
    np.random.seed(7)
    expected = np.random.random(5)
    np.random.seed(7)
    sample_circle_indices(500, 100, seed=0)
    assert np.array_equal(np.random.random(5), expected)


def test_empty_and_small_populations():
    assert len(sample_circle_indices(0, 100, seed=0)) == 0
    assert sorted(sample_circle_indices(3, 100, seed=0)) == [0, 1, 2]


@pytest.mark.parametrize("population,count,seed", [(-1, 1, 0), (1, -1, 0), (1, 1, -1)])
def test_invalid_parameters_fail_clearly(population, count, seed):
    with pytest.raises(ValueError):
        sample_circle_indices(population, count, seed=seed)
