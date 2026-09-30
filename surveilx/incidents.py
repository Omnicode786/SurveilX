import time

from fastapi import HTTPException
from sqlalchemy import select

from surveilx.database import Alert, Incident, audit

TRANSITIONS = {
    "VERIFYING": {"ACKNOWLEDGED", "FALSE_POSITIVE", "ESCALATED", "CONFIRMED", "EXPIRED"},
    "CONFIRMED": {"ALERTED", "ACKNOWLEDGED", "FALSE_POSITIVE", "ESCALATED"},
    "ALERTED": {"ACKNOWLEDGED", "FALSE_POSITIVE", "ESCALATED"},
    "ACKNOWLEDGED": {"IN_PROGRESS", "RESOLVED", "FALSE_POSITIVE", "ESCALATED"},
    "IN_PROGRESS": {"RESOLVED", "ESCALATED", "FALSE_POSITIVE"},
    "ESCALATED": {"ACKNOWLEDGED", "IN_PROGRESS", "RESOLVED", "FALSE_POSITIVE"},
    "RESOLVED": set(),
    "FALSE_POSITIVE": set(),
    "EXPIRED": set(),
}


def transition(session, incident, state, actor, note=""):
    if state not in TRANSITIONS.get(incident.state, set()):
        raise HTTPException(409, f"Cannot transition {incident.state} to {state}")
    before = incident.state
    incident.state = state
    incident.updated = time.time()
    audit(session, actor, "incident_transition", incident.id, old=before, new=state, note=note)


def create_incident(session, camera_id, evidence_key, details, synthetic=False):
    event_type = "synthetic_zone_entry" if synthetic else "zone_occupancy"
    policy = details.get("policy_observation")
    if policy:
        event_type = f"{'synthetic_' if synthetic else ''}rule:{policy['rule_id']}:{policy['kind']}"
    # A database commit may succeed before spool unlink. Both original and grouped
    # evidence keys must be idempotent, including observations of resolved incidents.
    for incident in session.scalars(select(Incident).where(Incident.camera_id == camera_id)).all():
        if evidence_key == incident.evidence_key or evidence_key in incident.details.get(
            "additional_evidence", []
        ):
            return incident
    now = time.time()
    existing = session.scalar(
        select(Incident)
        .where(
            Incident.camera_id == camera_id,
            Incident.event_type == event_type,
            Incident.evidence_key.is_not(None),
            Incident.created >= now - 300,
            Incident.state.in_(
                ["VERIFYING", "CONFIRMED", "ALERTED", "ACKNOWLEDGED", "IN_PROGRESS", "ESCALATED"]
            ),
        )
        .order_by(Incident.created.desc())
        .limit(1)
    )
    if existing and len(existing.details.get("additional_evidence", [])) < 99:
        existing.updated = now
        # Keep the original bundle and retain a reference to later observations.
        previous = existing.details
        bundles = previous.get("additional_evidence", [])
        if evidence_key not in bundles and evidence_key != existing.evidence_key:
            bundles = [*bundles, evidence_key]
        existing.details = previous | {
            "additional_evidence": bundles,
            "observations": previous.get("observations", 1) + 1,
            "latest_observation": details,
        }
        audit(session, "runtime", "observation_grouped", existing.id, evidence_key=evidence_key)
        return existing
    incident = Incident(
        camera_id=camera_id, event_type=event_type, evidence_key=evidence_key, details=details
    )
    session.add(incident)
    session.flush()
    session.add(Alert(incident_id=incident.id))
    audit(session, "runtime", "incident_created", incident.id, synthetic=synthetic)
    audit(session, "runtime", "notification_delivered_in_app", incident.id)
    return incident
