import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/compare_core_profiles.py"
spec = importlib.util.spec_from_file_location("compare_core_profiles", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_prediction():
    return {"predictions": [{"id": "0", "image_id": "a", "video_id": "a", "frame": 1,
                             "track_id": 2, "attributes": {"jersey": "10", "team": "left"},
                             "bbox_image": {"x": 12.5}}]}


def test_comparison_ignores_only_job_specific_ids():
    before = fixture_prediction()
    after = copy.deepcopy(before)
    for field in ("id", "image_id", "video_id"):
        after["predictions"][0][field] = "new job"
    assert module.compare_predictions(before, after)["exact_equal"]


@pytest.mark.parametrize("field,value", [
    ("track_id", 3), ("frame", 2), ("attributes", {"jersey": "11", "team": "left"}),
    ("bbox_image", {"x": 12.5000000001}),
])
def test_comparison_does_not_hide_geometry_identity_or_attribute_changes(field, value):
    before = fixture_prediction()
    after = copy.deepcopy(before)
    after["predictions"][0][field] = value
    assert not module.compare_predictions(before, after)["exact_equal"]


def test_duplicate_frame_track_rejected():
    document = fixture_prediction()
    document["predictions"].append(copy.deepcopy(document["predictions"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        module.compare_predictions(document, document)
