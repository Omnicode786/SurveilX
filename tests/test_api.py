import time
from types import SimpleNamespace


def login(client):
    assert (
        client.post(
            "/api/auth/login", json={"username": "admin", "password": "test-password-long"}
        ).status_code
        == 200
    )


def test_auth_csrf_rbac_and_rotation(client):
    assert client.get("/api/cameras").status_code == 401
    login(client)
    assert (
        client.post(
            "/api/users", json={"username": "viewer", "password": "viewer-password-long", "role": "viewer"}
        ).status_code
        == 200
    )
    assert client.post("/api/cameras", headers={"Origin": "https://evil.example"}, json={}).status_code == 403
    old = client.cookies.get("surveilx_session")
    assert client.post("/api/auth/refresh").status_code == 200
    assert old != client.cookies.get("surveilx_session")
    client.cookies.set("surveilx_session", old, domain="testserver.local", path="/")
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.clear()
    client.post("/api/auth/login", json={"username": "viewer", "password": "viewer-password-long"})
    assert client.post("/api/cameras", json={"name": "test", "source": "demo://0"}).status_code == 403


def test_incident_transitions_feedback_and_evidence(client):
    login(client)
    camera = client.post(
        "/api/cameras", json={"name": "Fixture", "source": "demo://0", "zones": [[0.6, 0.2, 0.96, 0.9]]}
    ).json()
    from surveilx.database import transaction
    from surveilx.evidence import save_bundle
    from surveilx.incidents import create_incident
    from surveilx.vision import generated_frame

    key = save_bundle([(time.time(), generated_frame(0, 1))], {"synthetic": True})
    with transaction() as session:
        incident = create_incident(session, camera["id"], key, {"synthetic": True}, True)
        incident_id = incident.id
    assert client.post(f"/api/incidents/{incident_id}/resolve").status_code == 409
    assert client.post(f"/api/incidents/{incident_id}/acknowledge").status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/resolve").status_code == 200
    assert client.post(f"/api/incidents/{incident_id}/dismiss").status_code == 409
    evidence = client.get(f"/api/incidents/{incident_id}/evidence")
    assert evidence.status_code == 200 and evidence.content.startswith(b"PK")
    feedback = client.post("/api/feedback", json={"incident_id": incident_id, "label": "true_event"}).json()
    assert client.post(f"/api/feedback/{feedback['id']}/review").status_code == 409
    assert any(a["action"] == "evidence_accessed" for a in client.get("/api/audit").json())
    client.delete(f"/api/cameras/{camera['id']}")


def test_input_rejects_path_and_box_errors(client):
    login(client)
    assert (
        client.post("/api/cameras", json={"name": "bad", "source": "C:/Windows/win.ini"}).status_code == 422
    )
    assert (
        client.post(
            "/api/cameras", json={"name": "bad", "source": "demo://0", "zones": [[0.9, 0, 0.1, 1]]}
        ).status_code
        == 422
    )


def test_deployment_rollback_is_targeted_durable_and_preserves_previous_on_promotion(client, monkeypatch):
    login(client)
    from surveilx import api
    from surveilx.database import ModelVersion, transaction
    from surveilx.runtime import Runtime
    from surveilx.acceptance import accept_report
    from surveilx.config import settings
    from test_acceptance import accepted_artifact

    def prepare(self, model):
        return SimpleNamespace(version=model.version, manifest=model.manifest)

    monkeypatch.setattr(Runtime, "prepare_model", prepare)
    with transaction() as session:
        first = ModelVersion(name="first", version="first", manifest={"deployment_eligible": True})
        second = ModelVersion(name="second", version="second", manifest={"deployment_eligible": True})
        session.add_all([first, second])
        session.flush()
        first_id, second_id = first.id, second.id
        report_id = accepted_artifact(session, second, settings.data_dir)
    accept_report(report_id, "test-reviewer")
    assert client.post(f"/api/models/{first_id}/deploy", json={"stage": "canary"}).status_code == 200
    assert client.post(f"/api/models/{second_id}/deploy", json={"stage": "canary"}).status_code == 200
    assert client.get(f"/api/models/{first_id}").json()["stage"] == "superseded"
    assert client.post(f"/api/models/{first_id}/rollback").status_code == 409
    assert client.post(f"/api/models/{second_id}/deploy", json={"stage": "production"}).status_code == 200
    restored = Runtime()
    restored.restore_models()
    assert restored.active_models["event"] == second_id
    assert restored.sva.version == "second"
    result = client.post(f"/api/models/{second_id}/rollback")
    assert result.status_code == 200 and result.json()["restored_model_id"] == first_id
    assert api.runtime.sva.version == "first"
    assert api.runtime.active_models["event"] == first_id
    assert client.get(f"/api/models/{first_id}").json()["stage"] == "canary"
    assert client.post(f"/api/models/{second_id}/rollback").status_code == 409


def test_failed_model_load_keeps_current_deployment(client, monkeypatch):
    login(client)
    from surveilx import api
    from surveilx.database import ModelVersion, transaction

    with transaction() as session:
        model = ModelVersion(name="missing", version="missing", manifest={"synthetic": True})
        session.add(model)
        session.flush()
        model_id = model.id
    assert client.post(f"/api/models/{model_id}/deploy", json={"stage": "production"}).status_code == 409
    assert client.post(f"/api/models/{model_id}/deploy", json={"stage": "canary"}).status_code == 409
    assert client.get(f"/api/models/{model_id}").json()["stage"] == "candidate"
    assert api.runtime.active_models == {"event": None, "detection": None}


def test_grouped_evidence_is_accessible_and_retry_does_not_duplicate_observation(client):
    login(client)
    camera = client.post("/api/cameras", json={"name": "Evidence", "source": "demo://0"}).json()
    from surveilx.database import transaction
    from surveilx.evidence import save_bundle
    from surveilx.incidents import create_incident

    first = save_bundle([], {"observation": 1})
    second = save_bundle([], {"observation": 2})
    with transaction() as session:
        incident = create_incident(session, camera["id"], first, {"synthetic": True}, True)
        incident_id = incident.id
    for _ in range(2):
        with transaction() as session:
            grouped = create_incident(session, camera["id"], second, {"synthetic": True}, True)
            assert grouped.id == incident_id
    details = client.get(f"/api/incidents/{incident_id}").json()["details"]
    assert details["observations"] == 2
    assert details["additional_evidence"] == [second]
    assert client.get(f"/api/incidents/{incident_id}/evidence?observation=1").status_code == 200
    assert client.get(f"/api/incidents/{incident_id}/evidence?observation=2").status_code == 404
    assert client.get(f"/api/incidents/{incident_id}/evidence?observation=-1").status_code == 422
