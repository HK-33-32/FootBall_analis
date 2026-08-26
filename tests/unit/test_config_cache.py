from football_intelligence.cache import InferenceCache
from football_intelligence.config import config_hash, load_config


def test_config_inheritance_and_hash_are_deterministic(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested:\n  left: 2\n  right: 3\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nnested:\n  right: 4\n", encoding="utf-8"
    )
    config = load_config(tmp_path / "child.yaml")
    assert config == {"a": 1, "nested": {"left": 2, "right": 4}}
    assert config_hash(config) == config_hash({"nested": {"right": 4, "left": 2}, "a": 1})


def test_inference_cache_is_content_addressed_and_atomic(tmp_path):
    cache = InferenceCache(tmp_path)
    key = cache.key({"model": "m", "revision": "sha", "frames": [1, 2]})
    assert cache.get(key) is None
    path = cache.put(key, {"primary_event": "unknown"})
    assert path.is_file()
    assert cache.get(key) == {"primary_event": "unknown"}
