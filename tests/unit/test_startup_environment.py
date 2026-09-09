import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "scripts/audit_startup_environment.py"
spec = importlib.util.spec_from_file_location("audit_startup_environment", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(*names):
    return {"timestamp": "test-only", "containers": [
        {"name": name, "image_id": "image", "started_at": "start"} for name in names]}


def test_excludes_only_named_benchmark_container():
    before = snapshot("/infra", "/fi-startup-baseline")
    after = snapshot("/infra")
    assert module.compare_snapshots(before, after, ["/fi-startup-baseline"])["same_service_set"]
    after["containers"].append(snapshot("/fi-startup-another-job")["containers"][0])
    assert not module.compare_snapshots(before, after, ["/fi-startup-baseline"])["same_service_set"]


def test_restart_or_image_change_is_detected():
    before, after = snapshot("/infra"), snapshot("/infra")
    after["containers"][0]["started_at"] = "new-start"
    result = module.compare_snapshots(before, after)
    assert not result["same_service_set"]
    assert result["changed"] == ["/infra"]
