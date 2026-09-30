import json
import time

import cv2
import numpy as np
import pytest


def detection_fixture(root, name="base", offset=0, synthetic=True):
    from training.datasets import digest

    directory = root / "datasets" / name
    directory.mkdir(parents=True)
    samples = []
    for i, split in enumerate(("train", "validation", "calibration", "test")):
        path = directory / f"{i}.png"
        cv2.imwrite(str(path), np.full((32, 32, 3), offset + i * 20, np.uint8))
        samples.append(
            {
                "image": path.name,
                "sha256": digest(path),
                "split": split,
                "group": f"{name}-{split}",
                "boxes": [[0.1, 0.1, 0.9, 0.9]],
                "labels": [0],
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "detection",
        "domain": "fixture",
        "classes": ["person"],
        "license": "Generated fixture",
        "synthetic": synthetic,
        "samples": samples,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory / "manifest.json"


def evidence(client):
    from surveilx.database import transaction
    from surveilx.evidence import save_bundle
    from surveilx.incidents import create_incident
    from surveilx.vision import generated_frame

    client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"})
    camera = client.post("/api/cameras", json={"name": "Annotations", "source": "demo://0"}).json()
    key = save_bundle([(time.time() + i, generated_frame(i * 2, 1)) for i in range(8)], {"synthetic": True})
    with transaction() as session:
        incident = create_incident(session, camera["id"], key, {"synthetic": True}, True)
        return {
            "incident_id": incident.id,
            "task": "detection",
            "domain": "fixture",
            "source_group": camera["id"],
            "rights": "Generated test evidence",
            "complete_annotation": True,
            "frames": [{"index": 0, "boxes": [{"box": [0.1, 0.2, 0.8, 0.9], "label": "person"}]}],
        }


def approved(client, body):
    from surveilx.adaptation import review_annotation

    response = client.post("/api/adaptation/annotations", json=body)
    assert response.status_code == 201, response.text
    record = response.json()
    review_annotation(record["id"], "independent-test-reviewer", True)
    return record["id"]


def test_annotation_review_geometry_and_access(client):
    body = evidence(client)
    listing = client.get(f"/api/incidents/{body['incident_id']}/frames").json()
    assert len(listing["frames"]) == 8
    assert client.get(listing["frames"][0]["url"]).content[:2] == b"\xff\xd8"
    record = client.post("/api/adaptation/annotations", json=body).json()
    path = f"/api/adaptation/annotations/{record['id']}/review"
    assert client.post(path, json={"approve": True}).status_code == 409
    client.post(
        "/api/users", json={"username": "reviewer", "password": "review-password-long", "role": "researcher"}
    )
    client.post("/api/auth/login", json={"username": "reviewer", "password": "review-password-long"})
    assert client.post(path, json={"approve": True}).json()["payload"]["state"] == "approved"
    assert client.post(path, json={"approve": False}).status_code == 409
    for change in (
        {"complete_annotation": False},
        {"frames": [{"index": 8, "boxes": []}]},
        {"frames": [{"index": 0, "boxes": [{"box": [0.8, 0, 0.1, 1], "label": "person"}]}]},
        {"task": "event", "event_label": "normal"},
    ):
        assert client.post("/api/adaptation/annotations", json={**body, **change}).status_code == 422


def test_dataset_generation_preserves_heldouts_and_prevents_reuse(client):
    from surveilx.config import settings
    from training.datasets import validate_manifest

    path = detection_fixture(settings.data_dir)
    body = evidence(client)
    annotation = approved(client, body)
    result = client.post(
        "/api/adaptation/datasets",
        json={"name": "generation2", "base_dataset": "base", "annotation_ids": [annotation]},
    )
    assert result.status_code == 201, result.text
    generated, counts = validate_manifest(settings.data_dir / "datasets/generation2/manifest.json")
    assert counts == {"train": 2, "validation": 1, "calibration": 1, "test": 1}
    base = json.loads(path.read_text())
    for old in base["samples"][1:]:
        new = next(s for s in generated["samples"] if s["split"] == old["split"])
        assert {k: v for k, v in new.items() if k != "image"} == {
            k: v for k, v in old.items() if k != "image"
        }
    assert generated["adaptation"]["annotations"][0]["reviewer"] == "independent-test-reviewer"
    assert (
        client.post(
            "/api/adaptation/datasets",
            json={"name": "generation3", "base_dataset": "generation2", "annotation_ids": [annotation]},
        ).status_code
        == 422
    )
    duplicate = approved(client, body)
    assert (
        client.post(
            "/api/adaptation/datasets",
            json={"name": "generation3", "base_dataset": "generation2", "annotation_ids": [duplicate]},
        ).status_code
        == 422
    )
    assert not (settings.data_dir / "datasets/generation3").exists()


@pytest.mark.parametrize("kind", ["pending", "taxonomy", "heldout", "scope"])
def test_invalid_annotation_cannot_publish_dataset(client, kind):
    from surveilx.config import settings

    detection_fixture(settings.data_dir)
    body = evidence(client)
    if kind == "taxonomy":
        body["frames"][0]["boxes"][0]["label"] = "unknown"
    if kind == "heldout":
        body["source_group"] = "base-test"
    if kind == "scope":
        body["domain"] = "wrong-domain"
    record = client.post("/api/adaptation/annotations", json=body).json()
    if kind != "pending":
        from surveilx.adaptation import review_annotation

        review_annotation(record["id"], "independent-test-reviewer", True)
    response = client.post(
        "/api/adaptation/datasets",
        json={"name": "invalid", "base_dataset": "base", "annotation_ids": [record["id"]]},
    )
    assert response.status_code == 422, response.text
    assert not (settings.data_dir / "datasets/invalid").exists()


def test_reviewed_event_export_matches_input_contract(client):
    from surveilx.config import settings
    from training.datasets import generate, validate_manifest

    base = settings.data_dir / "datasets/eventbase"
    generate(base, 80, seed=41)
    manifest = json.loads((base / "manifest.json").read_text())
    body = evidence(client)
    body.update(
        task="event",
        domain=manifest["domain"],
        event_label=manifest["classes"][0],
        frames=[
            {
                "index": i,
                "boxes": [
                    {"box": [0.1, 0.1, 0.4, 0.8], "label": "person", "track_id": 2},
                    {"box": [0.6, 0.1, 0.9, 0.8], "label": "person", "track_id": 1},
                ],
            }
            for i in range(8)
        ],
    )
    annotation = approved(client, body)
    response = client.post(
        "/api/adaptation/datasets",
        json={"name": "event2", "base_dataset": "eventbase", "annotation_ids": [annotation]},
    )
    assert response.status_code == 201, response.text
    dataset, counts = validate_manifest(settings.data_dir / "datasets/event2/manifest.json")
    assert counts["train"] == 21
    sample = next(s for s in dataset["samples"] if s.get("annotation_id") == annotation)
    with np.load(settings.data_dir / "datasets/event2" / sample["file"]) as data:
        assert data["clip"].shape == (8, 3, 64, 64)
        assert data["boxes"][0, 0, 0] == pytest.approx(0.6)  # Track 1 first, consistently.


def test_adaptation_worker_queues_once_and_advances_only_after_completion(client, monkeypatch):
    from surveilx import adaptation_worker as module
    from surveilx.config import settings
    from surveilx.database import Record, transaction

    detection_fixture(settings.data_dir)
    annotation = approved(client, evidence(client))
    worker = module.AdaptationWorker()
    worker.configure(module.AdaptationPolicy(base_dataset="base", minimum_approved=1), "admin")
    worker.tick()
    assert not worker.status().get("pending")
    calls = []

    def start(*args):
        calls.append(args)
        with transaction() as session:
            session.add(Record(id="training-fixture", kind="experiment", payload={"state": "running"}))
        return "training-fixture"

    monkeypatch.setattr(module.jobs, "start", start)
    worker.configure(module.AdaptationPolicy(base_dataset="base", minimum_approved=1, enabled=True), "admin")
    worker.tick()
    pending = worker.status()["pending"]
    worker.tick()
    assert len(calls) == 1 and pending["annotation_ids"] == [annotation]
    with transaction() as session:
        session.get(Record, "training-fixture").payload = {"state": "completed"}
    worker.tick()
    assert worker.status()["base_dataset"] == pending["dataset"]
    assert worker.status()["consumed_annotation_ids"] == [annotation]
    worker.tick()
    assert len(calls) == 1


def test_adaptation_failure_requires_explicit_retry(client, monkeypatch):
    from surveilx import adaptation_worker as module
    from surveilx.config import settings

    detection_fixture(settings.data_dir)
    approved(client, evidence(client))
    calls = []

    def fail(*args):
        calls.append(args)
        raise ValueError("test failure")

    monkeypatch.setattr(module, "build_dataset", fail)
    worker = module.AdaptationWorker()
    policy = module.AdaptationPolicy(base_dataset="base", minimum_approved=1, enabled=True)
    worker.configure(policy, "admin")
    worker.tick()
    worker.tick()
    assert len(calls) == 1 and worker.status()["state"] == "attention_required"
    worker.configure(policy, "admin")
    worker.tick()
    assert len(calls) == 2
