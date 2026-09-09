import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/compare_clip_loading.py"
spec = importlib.util.spec_from_file_location("compare_clip_loading", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def mock_run():
    return {"run_id": "test-only", "status": "complete", "load_s": 1.0,
            "checkpoint": {"sha256": "checkpoint", "missing": [], "unexpected": []},
            "source_manifest_sha256": "source", "script_sha256": "script", "config_hash": "cfg",
            "hardware": {}, "software": {}, "state": {"param": {"dtype": "fp16", "hash": "a"}},
            "inputs": ["real-in-production"], "outputs": {"logits": "a"},
            "requires_grad": {"param": False}, "training_flags": {"": False}}


def test_exact_comparison():
    before, after = mock_run(), mock_run()
    after["load_s"] = 0.5
    assert module.compare(before, after)["load_speedup"] == 2
    assert module.compare(before, after)["all_exact"]


@pytest.mark.parametrize("key,value", [
    ("state", {"param": {"dtype": "fp32", "hash": "a"}}),
    ("inputs", ["different"]), ("outputs", {"logits": "different"}),
    ("requires_grad", {"param": True}), ("training_flags", {"": True}),
])
def test_state_dtype_and_behavior_drift_is_not_hidden(key, value):
    before, after = mock_run(), mock_run()
    after[key] = value
    assert not module.compare(before, after)["all_exact"]


def test_partial_checkpoint_rejected():
    before, after = mock_run(), mock_run()
    after["checkpoint"]["missing"] = ["visual.conv1.weight"]
    with pytest.raises(ValueError, match="complete model state"):
        module.compare(before, after)


def test_wrong_checkpoint_rejected():
    before = mock_run()
    after = copy.deepcopy(before)
    after["checkpoint"]["sha256"] = "other"
    with pytest.raises(ValueError, match="differs"):
        module.compare(before, after)
