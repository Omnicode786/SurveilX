import pytest

from surveilx.task_profiles import coverage, declared_profiles, validate_profiles
from surveilx.task_profiles import development_evidence


def test_legacy_pedestrian_continuation_keeps_its_declared_family():
    metadata = {"task": "detection", "domain": "pedestrian", "classes": ["person"], "capability_ids": []}
    assert declared_profiles(metadata) == ["person_detection"]
    assert declared_profiles({**metadata, "domain": "retail"}) == []
    assert declared_profiles({**metadata, "task": "event"}) == []


def test_profiles_require_explicit_taxonomy_and_matching_task():
    metadata = {"task": "detection", "classes": ["fire", "smoke"], "capability_ids": ["fire_smoke"]}
    assert validate_profiles(metadata) == ["fire_smoke"]
    for change in (
        {"classes": ["fire"]},
        {"task": "event"},
        {"capability_ids": ["missing"]},
        {"capability_ids": ["fire_smoke", "fire_smoke"]},
        {"capability_ids": ["ppe_compliance"]},
    ):
        with pytest.raises(ValueError):
            validate_profiles({**metadata, **change})


def test_coverage_never_promotes_class_names_or_unverified_approval():
    metadata = {
        "task": "detection",
        "classes": ["fire", "smoke"],
        "domain": "warehouse",
        "acceptance": {"report_id": "forged"},
    }
    model = {"id": "candidate", "version": "v1", "stage": "production", "manifest": metadata}
    assert not next(p for p in coverage([model], []) if p["id"] == "fire_smoke")["models"]
    metadata["capability_ids"] = ["fire_smoke"]
    item = next(p for p in coverage([model], [], ["candidate"]) if p["id"] == "fire_smoke")
    assert item["models"][0]["active"]
    assert not item["production_validated"]
    item = next(p for p in coverage([model], [], (), ["candidate"]) if p["id"] == "fire_smoke")
    assert item["production_validated"]
    metadata["synthetic"] = True
    assert not next(p for p in coverage([model], [], (), ["candidate"]) if p["id"] == "fire_smoke")[
        "production_validated"
    ]


def test_coverage_endpoint_requires_authentication_and_reports_all_families(client):
    assert client.get("/api/capabilities").status_code == 401
    client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"})
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    profiles = {p["id"]: p for p in response.json()["profiles"]}
    assert {
        "fire_smoke",
        "firearm",
        "fighting",
        "fall",
        "theft",
        "traffic_collision",
        "traffic_violation",
        "ppe_compliance",
        "industrial_hazards",
    } <= profiles.keys()
    assert all(not p["production_validated"] for p in profiles.values())


def test_coverage_exposes_weak_staged_data_without_claiming_a_trained_model():
    metadata = {"task": "event", "classes": ["normal", "fighting"], "capability_ids": ["fighting"],
                "license": "research only", "provenance": {"label_scope": "inherited_video_label", "staged": True}}
    profile = next(p for p in coverage([], [("staged", metadata)]) if p["id"] == "fighting")
    assert profile["status"] == "data_available"
    assert not profile["models"] and not profile["production_validated"]
    assert profile["dataset_evidence"][0]["provenance"]["staged"]


def test_aggregate_score_never_hides_weak_or_untested_classes():
    metadata = {"task": "detection", "classes": ["fire", "smoke", "untested"],
                "metrics": {"map50": 0.6, "per_class": [
                    {"class_id": 0, "ground_truth": 20, "ap50": 0.8},
                    {"class_id": 1, "ground_truth": 20, "ap50": 0.4}]}}
    result = development_evidence(metadata)
    assert result["aggregate_target_met"]
    assert not result["all_classes_target_met"]
    assert result["per_class"][2]["score"] is None
    assert not result["independent_reliability_established"]


def test_generation_status_detects_stale_runner_and_preserves_journal(client, monkeypatch):
    import json
    import psutil
    from surveilx.config import settings

    assert client.get("/api/generations").status_code == 401
    client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"})
    directory = settings.data_dir / "generations" / "fixture"
    directory.mkdir(parents=True)
    path = directory / "status.json"
    path.write_text(json.dumps({"state": "running", "pid": 1, "jobs": {}, "config": {"epochs": 3}}))

    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing)
    response = client.get("/api/generations")
    assert response.status_code == 200
    assert response.json()[0]["state"] == "interrupted"
    assert json.loads(path.read_text())["state"] == "running"
