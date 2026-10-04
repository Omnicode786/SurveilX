"""Durable in-app notification policy, grouping, acknowledgement and escalation."""

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from surveilx.database import Alert, Incident, Record, audit, serialize

Role = Literal["admin", "operator", "researcher", "analyst", "viewer"]
Priority = Literal["review", "low", "medium", "high", "critical"]
TERMINAL = {"RESOLVED", "FALSE_POSITIVE", "EXPIRED"}
REVIEWED = TERMINAL | {"ACKNOWLEDGED", "IN_PROGRESS"}


class NotificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    incident_group_seconds: int = Field(default=300, ge=0, le=3600)
    cooldown_seconds: int = Field(default=60, ge=0, le=3600)
    acknowledgement_seconds: int = Field(default=0, ge=0, le=86400)
    recipient_roles: list[Role] = Field(
        default=["admin", "operator", "researcher", "analyst", "viewer"], min_length=1, max_length=5
    )
    escalation_roles: list[Role] = Field(default=["admin"], min_length=1, max_length=5)
    event_priorities: dict[str, Priority] = Field(default_factory=dict, max_length=64)

    @field_validator("event_priorities")
    @classmethod
    def valid_event_names(cls, value):
        if any(not key.strip() or key != key.strip() or len(key) > 80 for key in value):
            raise ValueError("Event identifiers must be nonempty, trimmed and at most 80 characters")
        return value


def get_policy(session):
    row = session.get(Record, "notification-policy")
    return NotificationPolicy.model_validate(row.payload if row else {})


def put_policy(session, value, actor):
    payload = value.model_dump()
    row = session.get(Record, "notification-policy")
    before = row.payload if row else NotificationPolicy().model_dump()
    if row:
        row.payload, row.timestamp = payload, time.time()
    else:
        session.add(Record(id="notification-policy", kind="notification_policy", payload=payload))
    audit(session, actor, "notification_policy_changed", "notification-policy", old=before, new=payload)
    return payload


def notify(session, incident, now=None):
    now = time.time() if now is None else now
    policy = get_policy(session)
    if session.scalar(select(Alert).where(Alert.incident_id == incident.id)):
        return
    if not policy.enabled:
        audit(session, "runtime", "notification_suppressed", incident.id, reason="policy_disabled")
        return
    key = f"{incident.camera_id}:{incident.event_type}"
    recent = session.scalars(
        select(Record)
        .where(Record.kind == "notification", Record.timestamp >= now - policy.cooldown_seconds)
        .order_by(Record.timestamp.desc())
        .limit(500)
    )
    for record in recent:
        payload = record.payload
        alert = session.get(Alert, payload["alert_id"])
        if (
            payload["group_key"] == key
            and alert
            and alert.status in {"delivered_in_app", "escalated"}
            and len(payload["incident_ids"]) < 100
        ):
            if incident.id not in payload["incident_ids"]:
                record.payload = {**payload, "incident_ids": [*payload["incident_ids"], incident.id]}
                audit(session, "runtime", "notification_grouped", alert.id, incident_id=incident.id)
            return
    alert = Alert(incident_id=incident.id, created=now)
    session.add(alert)
    session.flush()
    payload = {
        "alert_id": alert.id,
        "group_key": key,
        "incident_ids": [incident.id],
        "event_type": incident.event_type,
        "camera_id": incident.camera_id,
        "priority": policy.event_priorities.get(incident.event_type, incident.severity),
        "recipient_roles": policy.recipient_roles,
        "escalation_roles": policy.escalation_roles,
        "delivered_at": now,
        "acknowledgement_due": now + policy.acknowledgement_seconds
        if policy.acknowledgement_seconds
        else None,
        "escalations": 0,
        "delivery_scope": "Persisted in-app; external channels are unconfigured",
    }
    session.add(Record(id=f"notification:{alert.id}", kind="notification", timestamp=now, payload=payload))
    audit(
        session,
        "runtime",
        "notification_delivered_in_app",
        incident.id,
        alert_id=alert.id,
        priority=payload["priority"],
        recipient_roles=payload["recipient_roles"],
    )


def acknowledge(session, alert, actor, now=None):
    if alert.status == "closed":
        raise ValueError("This notification is closed")
    if alert.status == "acknowledged":
        return
    now = time.time() if now is None else now
    alert.acknowledged_by, alert.status = actor, "acknowledged"
    record = session.get(Record, f"notification:{alert.id}")
    if record:
        record.payload = {**record.payload, "acknowledged_at": now, "acknowledged_by": actor}
    audit(session, actor, "alert_acknowledged", alert.id)


def reconcile_incident(session, incident, actor):
    for row in session.scalars(
        select(Record).where(
            Record.kind == "notification",
            Record.payload["incident_ids"].as_string().contains(f'"{incident.id}"', autoescape=True),
        )
    ):
        if incident.id not in row.payload["incident_ids"]:
            continue
        alert = session.get(Alert, row.payload["alert_id"])
        if not alert or alert.status in {"closed", "acknowledged"}:
            continue
        states = [
            (item.state if (item := session.get(Incident, key)) else "EXPIRED")
            for key in row.payload["incident_ids"]
        ]
        if all(state in TERMINAL for state in states):
            alert.status = "closed"
            row.payload = {**row.payload, "closed_at": time.time()}
            audit(session, actor, "notification_closed", alert.id)
        elif all(state in REVIEWED for state in states):
            acknowledge(session, alert, actor)


def process_due(session, now=None):
    now = time.time() if now is None else now
    count = 0
    records = session.scalars(
        select(Record)
        .where(
            Record.kind == "notification",
            Record.payload["acknowledgement_due"].as_float() <= now,
            Record.payload["escalations"].as_integer() == 0,
        )
        .join(Alert, Alert.id == Record.payload["alert_id"].as_string())
        .where(Alert.status == "delivered_in_app")
        .order_by(Record.timestamp)
        .limit(500)
    )
    for record in records:
        alert = session.get(Alert, record.payload["alert_id"])
        payload = record.payload
        due = payload.get("acknowledgement_due")
        if due is None or now < due or payload["escalations"]:
            continue
        incidents = [session.get(Incident, key) for key in payload["incident_ids"]]
        if all(not item or item.state in TERMINAL for item in incidents):
            alert.status = "closed"
            record.payload = {**payload, "closed_at": now}
            audit(session, "notification_engine", "notification_closed", alert.id)
            continue
        if all(not item or item.state in REVIEWED for item in incidents):
            acknowledge(session, alert, "notification_engine", now)
            continue
        # Escalation requests attention; it does not confirm the perceived event.
        alert.status = "escalated"
        record.payload = {
            **payload,
            "escalations": 1,
            "escalated_at": now,
            "recipient_roles": sorted(set(payload["recipient_roles"] + payload["escalation_roles"])),
        }
        audit(
            session,
            "notification_engine",
            "notification_escalated",
            alert.id,
            incident_ids=payload["incident_ids"],
            recipient_roles=payload["escalation_roles"],
        )
        count += 1
    return count


def visible_alerts(session, role):
    result = []
    for alert in session.scalars(select(Alert).order_by(Alert.created.desc()).limit(1000)):
        record = session.get(Record, f"notification:{alert.id}")
        payload = record.payload if record else {}
        if payload and role not in payload["recipient_roles"]:
            continue
        result.append({**serialize(alert), "notification": payload})
    return result
