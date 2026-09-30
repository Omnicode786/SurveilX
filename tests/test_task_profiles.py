import pytest

from surveilx.task_profiles import coverage, validate_profiles


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
