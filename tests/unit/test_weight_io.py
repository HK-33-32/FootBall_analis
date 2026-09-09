import hashlib
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/benchmark_weight_io.py"
spec = importlib.util.spec_from_file_location("benchmark_weight_io", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

MOCK_UPSTREAM = '''
def _compute_file_md5(filepath: str) -> str:
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
'''


@pytest.mark.parametrize("size", [0, 4095, 4096, 4097, 1048577])
def test_larger_blocks_preserve_complete_digest(tmp_path, size):
    weight = tmp_path / "test-only.bin"
    value = (bytes(range(256)) * (size // 256 + 1))[:size]
    weight.write_bytes(value)
    candidate = module.load_function(module.optimized_source(MOCK_UPSTREAM, 1048576))
    assert candidate(weight) == hashlib.md5(value).hexdigest()
    # No stale result reuse, including equal-sized mutation.
    weight.write_bytes(b"x" * size)
    assert candidate(weight) == hashlib.md5(b"x" * size).hexdigest()


@pytest.mark.parametrize("size", [0, -1, 4095, True, 1.5])
def test_invalid_blocks_rejected(size):
    with pytest.raises(ValueError):
        module.optimized_source(MOCK_UPSTREAM, size)


def test_upstream_drift_rejected():
    with pytest.raises(ValueError, match="upstream"):
        module.optimized_source(MOCK_UPSTREAM.replace("read(4096)", "read(8192)"), 1048576)


def test_missing_weight_fails(tmp_path):
    candidate = module.load_function(module.optimized_source(MOCK_UPSTREAM, 1048576))
    with pytest.raises(FileNotFoundError):
        candidate(tmp_path / "missing")
