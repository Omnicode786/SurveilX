from sqlalchemy import select


def setup_policy(session, **options):
    from surveilx.database import Camera
    from surveilx.notifications import NotificationPolicy, put_policy

    camera = Camera(name="Review camera", source_cipher="fixture")
    session.add(camera)
    session.flush()
    put_policy(session, NotificationPolicy(**options), "admin")
    return camera


def incident(session, camera, now, event="fire"):
    from surveilx.database import Incident
    from surveilx.notifications import notify

    item = Incident(camera_id=camera.id, event_type=event, created=now, updated=now)
    session.add(item)
    session.flush()
    notify(session, item, now)
    return item


def test_grouping_and_escalation_preserve_unconfirmed_incidents(client):
    from surveilx.database import Alert, Record, transaction
    from surveilx.notifications import process_due, visible_alerts

    with transaction() as session:
        camera = setup_policy(
            session,
            recipient_roles=["operator"],
            acknowledgement_seconds=10,
            event_priorities={"fire": "critical"},
        )
        first = incident(session, camera, 100)
        second = incident(session, camera, 101)
        other = incident(session, camera, 102, "smoke")
        session.flush()
        assert len(list(session.scalars(select(Alert)))) == 2
        assert visible_alerts(session, "viewer") == []
        assert process_due(session, 109) == 0
        assert process_due(session, 110) == 1
        assert process_due(session, 110) == 0
        alert = session.scalar(select(Alert).where(Alert.incident_id == first.id))
        payload = session.get(Record, f"notification:{alert.id}").payload
        assert payload["incident_ids"] == [first.id, second.id]
        assert payload["priority"] == "critical"
        assert payload["recipient_roles"] == ["admin", "operator"]
        assert first.state == second.state == other.state == "VERIFYING"
        assert len(visible_alerts(session, "admin")) == 1


def test_group_review_requires_all_incidents_and_stops_escalation(client):
    from surveilx.database import Alert, transaction
    from surveilx.notifications import process_due, reconcile_incident

    with transaction() as session:
        camera = setup_policy(session, acknowledgement_seconds=10)
        first = incident(session, camera, 100)
        second = incident(session, camera, 101)
        first.state = "FALSE_POSITIVE"
        reconcile_incident(session, first, "operator")
        alert = session.scalar(select(Alert).where(Alert.incident_id == first.id))
        assert alert.status == "delivered_in_app"
        second.state = "RESOLVED"
        reconcile_incident(session, second, "operator")
        assert alert.status == "closed"
        assert process_due(session, 200) == 0


def test_acknowledgement_and_disabled_policy(client):
    from surveilx.database import Alert, transaction
    from surveilx.notifications import NotificationPolicy, acknowledge, process_due, put_policy

    with transaction() as session:
        camera = setup_policy(session, acknowledgement_seconds=10)
        first = incident(session, camera, 100)
        alert = session.scalar(select(Alert).where(Alert.incident_id == first.id))
        acknowledge(session, alert, "operator", 105)
        assert process_due(session, 200) == 0
        put_policy(session, NotificationPolicy(enabled=False), "admin")
        incident(session, camera, 201)
        assert len(list(session.scalars(select(Alert)))) == 1


def test_legacy_alerts_do_not_starve_due_notifications(client):
    from surveilx.database import Alert, Incident, transaction
    from surveilx.notifications import process_due

    with transaction() as session:
        camera = setup_policy(session, acknowledgement_seconds=10)
        for _ in range(501):
            item = Incident(camera_id=camera.id, event_type="legacy", created=1)
            session.add(item)
            session.flush()
            session.add(Alert(incident_id=item.id, created=1))
        incident(session, camera, 100)
        session.flush()
        assert process_due(session, 110) == 1


def test_policy_api_is_admin_only_and_rejects_invalid_routes(client):
    from surveilx.database import User, transaction
    from surveilx.security import hasher

    assert (
        client.post(
            "/api/auth/login", json={"username": "admin", "password": "test-password-long"}
        ).status_code
        == 200
    )
    policy = client.get("/api/notifications/policy").json()
    assert client.put("/api/notifications/policy", json={**policy, "recipient_roles": []}).status_code == 422
    assert (
        client.put("/api/notifications/policy", json={**policy, "recipient_roles": ["operator"]}).status_code
        == 200
    )
    with transaction() as session:
        session.add(User(username="op", role="operator", password_hash=hasher.hash("test-password-long")))
    assert (
        client.post("/api/auth/login", json={"username": "op", "password": "test-password-long"}).status_code
        == 200
    )
    assert client.put("/api/notifications/policy", json=policy).status_code == 403


def test_live_and_rest_routing_match_and_unrouted_ack_is_forbidden(client):
    from surveilx.config import settings
    from surveilx.database import Alert, transaction

    assert (
        client.post(
            "/api/auth/login", json={"username": "admin", "password": "test-password-long"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/users",
            json={"username": "operator", "password": "operator-password-long", "role": "operator"},
        ).status_code
        == 200
    )
    with transaction() as session:
        camera = setup_policy(session, recipient_roles=["admin"])
        incident(session, camera, 100)
        alert_id = session.scalar(select(Alert)).id
    assert len(client.get("/api/alerts").json()) == 1
    with client.websocket_connect("/live/alerts", headers={"Origin": settings.allowed_origin}) as socket:
        assert len(socket.receive_json()["data"]) == 1
    client.cookies.clear()
    assert (
        client.post(
            "/api/auth/login", json={"username": "operator", "password": "operator-password-long"}
        ).status_code
        == 200
    )
    assert client.get("/api/alerts").json() == []
    assert client.post(f"/api/alerts/{alert_id}/acknowledge").status_code == 403
    with client.websocket_connect("/live/alerts", headers={"Origin": settings.allowed_origin}) as socket:
        assert socket.receive_json()["data"] == []
