"""Validate the existing ViT-L/14 checkpoint before constructing its CLIP backbone.

No tensors are fabricated and no weight fallback/download is performed here.
The upstream CLIP builder and full model load subsequently enforce all remaining keys.
"""

from collections.abc import Mapping

SUPPORTED_MODEL = "ViT-L/14"
EXPECTED_SHAPES = {
    "visual.conv1.weight": (1024, 3, 14, 14),
    "visual.positional_embedding": (257, 1024),
    "visual.proj": (1024, 768),
    "positional_embedding": (77, 768),
    "token_embedding.weight": (49408, 768),
    "ln_final.weight": (768,),
    "text_projection": (768, 768),
}


def clip_state_from_checkpoint(state, model_name=SUPPORTED_MODEL):
    """Return a non-mutating prefix view; reject drift from the supported architecture."""
    if model_name != SUPPORTED_MODEL:
        raise ValueError(
            "Direct loading supports ViT-L/14 only; use the legacy constructor "
            "explicitly for a different backbone"
        )
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ValueError("Expected a plain CLIPFinetune state dictionary")
    prefix = "clip_model."
    backbone = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    for name, expected in EXPECTED_SHAPES.items():
        value = backbone.get(name)
        if value is None or tuple(getattr(value, "shape", ())) != expected:
            raise ValueError(
                f"Checkpoint architecture mismatch: {prefix}{name}, expected {expected}"
            )
    for name, expected in (("visual.transformer.resblocks.", 24), ("transformer.resblocks.", 12)):
        actual = sum(
            key.startswith(name) and key.endswith(".attn.in_proj_weight") for key in backbone
        )
        if actual != expected:
            raise ValueError(f"Checkpoint transformer depth mismatch: {name}, expected {expected}")
    return backbone
