import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/compare_core_startup.py"
spec = importlib.util.spec_from_file_location("compare_core_startup", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def mock_run():
    return {"run_id": "test-only", "status": "complete", "video_sha256": "video",
            "frame_hashes": {"1.jpg": "frame"}, "source_config_sha256": "config",
            "split": "development", "loaded_weight_hashes": {"weight": "hash"},
            "profile_init": False, "outputs": {"tracks.txt": "output"},
            "hardware": {"gpu": "test-only"}, "software": {}, "script_sha256": "script",
            "source_hashes": {"file.py": "hash"},
            "stages": {"imports": 1.0, "initialization": 2.0, "tracking": 3.0}}


def test_exact_startup_comparison():
    before = mock_run()
    after = copy.deepcopy(before)
    after["stages"]["initialization"] = 1.0
    result = module.compare(before, after)
    assert result["exact_tracking_output"]
    assert result["timings"]["initialization"]["speedup"] == 2.0
    assert result["timed_tracking_stages"]["speedup"] == 1.2


@pytest.mark.parametrize("key,value", [
    ("video_sha256", "different"), ("frame_hashes", {}),
    ("source_config_sha256", "different"), ("split", "test"),
    ("loaded_weight_hashes", {}), ("profile_init", True), ("status", "failed"),
    ("hardware", {}), ("software", {"torch": "different"}), ("script_sha256", "different"),
])
def test_incomparable_startup_rejected(key, value):
    before, after = mock_run(), mock_run()
    after[key] = value
    with pytest.raises(ValueError):
        module.compare(before, after)


def test_changed_output_is_not_hidden():
    before, after = mock_run(), mock_run()
    after["outputs"]["tracks.txt"] = "different"
    assert not module.compare(before, after)["exact_tracking_output"]


def test_missing_output_is_not_exact():
    before, after = mock_run(), mock_run()
    before["outputs"] = after["outputs"] = {}
    with pytest.raises(ValueError, match="missing"):
        module.compare(before, after)


def test_aggregate_retains_samples_and_detects_between_pair_drift():
    before, after = [mock_run(), mock_run()], [mock_run(), mock_run()]
    after[0]["stages"]["initialization"] = 1.0
    after[1]["stages"]["initialization"] = 0.5
    result = module.aggregate(before, after)
    assert result["all_tracking_outputs_identical"]
    assert result["timings"]["initialization"]["candidate_median_s"] == 0.75
    before[1]["outputs"]["tracks.txt"] = after[1]["outputs"]["tracks.txt"] = "drift"
    result = module.aggregate(before, after)
    assert not result["all_tracking_outputs_identical"]


def test_empty_aggregate_rejected():
    with pytest.raises(ValueError):
        module.aggregate([], [])


def test_explicit_redundant_weight_removal_is_narrow():
    before, after = mock_run(), mock_run()
    before["loaded_weight_hashes"]["base_clip"] = "base-hash"
    with pytest.raises(ValueError, match="loaded_weight_hashes"):
        module.compare(before, after)
    result = module.compare(before, after, ["base_clip"])
    assert result["explicitly_removed_weight_files"] == ["base_clip"]
    after["loaded_weight_hashes"]["weight"] = "changed-finetuned-hash"
    with pytest.raises(ValueError, match="loaded_weight_hashes"):
        module.compare(before, after, ["base_clip"])


def test_new_weight_is_never_hidden_by_removal_allowance():
    before, after = mock_run(), mock_run()
    after["loaded_weight_hashes"]["new"] = "new-hash"
    with pytest.raises(ValueError, match="loaded_weight_hashes"):
        module.compare(before, after, ["new"])
