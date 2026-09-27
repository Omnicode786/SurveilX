import importlib
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import surveilx.config
    import surveilx.database

    monkeypatch.setattr(surveilx.config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(surveilx.config.settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(surveilx.config.settings, "admin_password", "test-password-long")
    monkeypatch.setattr(surveilx.config.settings, "start_workers", False)
    importlib.reload(surveilx.database)
    import surveilx.security
    import surveilx.incidents
    import surveilx.runtime
    import surveilx.api

    for module in [surveilx.security, surveilx.incidents, surveilx.runtime, surveilx.api]:
        importlib.reload(module)
    with TestClient(surveilx.api.app) as client:
        yield client


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
