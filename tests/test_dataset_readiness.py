import hashlib
import json

import cv2
import numpy as np
import pytest

from training.dataset_readiness import inspect_dataset


def detection_fixture(directory):
    directory.mkdir(parents=True)
    samples = []
    for index in range(12):
        split = ("train", "validation", "calibration", "test")[index // 3]
        name = f"image-{index}.png"
        cv2.imwrite(str(directory / name), np.full((32, 48, 3), index * 17, dtype=np.uint8))
        labels = [1] if index == 3 else [] if index == 6 else [0]
        samples.append({"image": name, "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                        "group": f"group-{index}", "split": split, "labels": labels,
                        "boxes": [[.1, .2, .8, .9]] if labels else []})
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "task": "detection", "domain": "fixture",
                               "license": "Generated test fixture", "synthetic": True,
                               "classes": ["person", "helmet", "rare"], "samples": samples}))
    return path


def test_readiness_counts_sparse_missing_and_empty_without_changing_data(tmp_path):
    path = detection_fixture(tmp_path / "dataset")
    original = path.read_bytes()
    report = inspect_dataset(path)
    assert report["valid"] and not report["model_quality_verified"]
    assert report["count_unit"] == "objects"
    assert report["per_class"][0]["labeled_instances"]["train"] == 3
    assert report["per_class"][1]["labeled_instances"]["validation"] == 1
    assert report["empty_samples"]["calibration"] == 1
    assert any(row["kind"] == "missing_supervision" and row["label"] == "rare" for row in report["issues"])
    assert path.read_bytes() == original
    (path.parent / "image-0.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        inspect_dataset(path)


def test_readiness_supports_video_event_contract(tmp_path):
    from training.datasets import generate
    generate(tmp_path / "events", count=80)
    report = inspect_dataset(tmp_path / "events/manifest.json")
    assert report["count_unit"] == "clips" and report["counts"]["train"] == 20
    assert sum(row["labeled_instances"]["train"] for row in report["per_class"]) == 20


def test_readiness_api_requires_researcher_and_rejects_bad_paths(client):
    from surveilx.config import settings
    assert client.get("/api/datasets/fixture/readiness").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"}).status_code == 200
    detection_fixture(settings.data_dir / "datasets/fixture")
    response = client.get("/api/datasets/fixture/readiness")
    assert response.status_code == 200 and response.json()["valid"]
    assert client.get("/api/datasets/bad%24name/readiness").status_code == 422
    assert client.get("/api/datasets/missing/readiness").status_code == 422
    assert client.post("/api/users", json={"username": "coverage-viewer", "password": "viewer-password-long", "role": "viewer"}).status_code == 200
    client.cookies.clear()
    client.post("/api/auth/login", json={"username": "coverage-viewer", "password": "viewer-password-long"})
    assert client.get("/api/datasets/fixture/readiness").status_code == 403
