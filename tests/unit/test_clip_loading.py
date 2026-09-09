from dataclasses import dataclass

import pytest

from football_intelligence.clip_loading import EXPECTED_SHAPES, clip_state_from_checkpoint


@dataclass
class MockTensorShape:
    shape: tuple


def mock_checkpoint():
    result = {"clip_model." + key: MockTensorShape(shape) for key, shape in EXPECTED_SHAPES.items()}
    for name, count in (("visual.transformer.resblocks", 24), ("transformer.resblocks", 12)):
        for index in range(count):
            result[f"clip_model.{name}.{index}.attn.in_proj_weight"] = MockTensorShape((1,))
    result["role_classifier.0.weight"] = MockTensorShape((256, 768))
    return result


def test_prefix_view_does_not_mutate_or_copy_tensors():
    state = mock_checkpoint()
    original_keys = set(state)
    backbone = clip_state_from_checkpoint(state)
    assert set(state) == original_keys
    assert "role_classifier.0.weight" not in backbone
    assert backbone["visual.conv1.weight"] is state["clip_model.visual.conv1.weight"]


def test_other_architecture_is_explicitly_rejected():
    with pytest.raises(ValueError, match="ViT-L/14 only"):
        clip_state_from_checkpoint(mock_checkpoint(), "ViT-B/32")


@pytest.mark.parametrize("key", list(EXPECTED_SHAPES))
def test_incomplete_backbone_is_rejected(key):
    state = mock_checkpoint()
    del state["clip_model." + key]
    with pytest.raises(ValueError, match="architecture mismatch"):
        clip_state_from_checkpoint(state)


def test_transformer_depth_drift_is_rejected():
    state = mock_checkpoint()
    del state["clip_model.visual.transformer.resblocks.23.attn.in_proj_weight"]
    with pytest.raises(ValueError, match="depth mismatch"):
        clip_state_from_checkpoint(state)
