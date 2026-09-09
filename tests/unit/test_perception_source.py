import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "services/perception"


def test_integrated_perception_has_required_runtime_and_portable_paths():
    required = (
        "core/api.py", "core/jobs.py", "engine/inference_soccernetGSR.py",
        "engine/fi_calibration_sampling.py", "engine/fi_clip_loading.py",
        "engine/prtreid/data/__init__.py", "engine/reid/torchreid/data/__init__.py",
        "engine/yolox/data/__init__.py",
        "engine/sfr/template/soccernet_template_97.npy", "scripts/entrypoint.sh",
    )
    assert all((SERVICE / name).is_file() for name in required)
    config = (SERVICE / "engine/configs/config.yaml").read_text(encoding="utf-8")
    assert ":/Users/" not in config and ":\\Users\\" not in config
    assert "checkpoints/Qwen2.5-VL-7B-Instruct-Q8_0.gguf" in config
    assert not list(SERVICE.rglob("*.pyc"))
    assert not list(SERVICE.rglob("*.so"))
    assert not list(SERVICE.rglob(".git"))


def test_compose_builds_integrated_source_and_mounts_weights():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    perception = compose["services"]["perception"]
    assert perception["build"]["dockerfile"] == "services/perception/Dockerfile"
    assert perception["image"].endswith("football-intelligence-perception:0.1.0}")
    assert any(volume.endswith(":/opt/weights") for volume in perception["volumes"])
    assert perception["environment"]["FG_CHECKPOINT_DIR"] == "/opt/weights/checkpoints"


def test_source_provenance_inventory_matches_repository():
    spec = importlib.util.spec_from_file_location(
        "verify_perception_source", ROOT / "scripts/verify_perception_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = json.loads((SERVICE / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["source_image_id"] == module.EXPECTED_IMAGE_ID
    assert provenance["inventory"] == module.local_inventory(SERVICE)
    assert provenance["files"] == provenance["exact_files"] + 4


def test_source_dockerfile_never_copies_weights_or_legacy_core_image():
    source = (SERVICE / "Dockerfile").read_text(encoding="utf-8")
    assert "football-core:1.0.0" not in source
    assert "COPY weights" not in source
    assert 'VOLUME ["/data", "/opt/weights"]' in source
    assert "FROM ${PERCEPTION_STAGE} AS final" in source


def test_checkpoint_report_rejects_wrong_model_revision(tmp_path):
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SERVICE))
    try:
        spec = importlib.util.spec_from_file_location(
            "fetch_weights", SERVICE / "scripts/fetch_weights.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SERVICE))
        sys.dont_write_bytecode = previous_bytecode
    for name in module.REQUIRED:
        (tmp_path / name).write_bytes(b"wrong revision")
    assert set(module.report(tmp_path)) == set(module.REQUIRED)
