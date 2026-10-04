import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, text

from surveilx.accelerators import capabilities
from surveilx.policies import SitePolicy
from surveilx.adaptation import (
    AnnotationInput,
    DatasetBuild,
    build_dataset,
    evidence_frames,
    review_annotation,
    submit_annotation,
)
from surveilx.adaptation_worker import AdaptationPolicy, adaptation_worker
from surveilx.acceptance import AcceptanceInput, accept_report, evaluation_jobs, verify_approved_artifact
from surveilx.config import settings
from surveilx.database import (
    Alert,
    Audit,
    Base,
    Camera,
    Feedback,
    Incident,
    ModelVersion,
    Record,
    User,
    audit,
    engine,
    serialize,
    transaction,
)
from surveilx.evidence import read_bundle
from surveilx.drift import drift_monitor
from surveilx.incidents import transition
from surveilx.notifications import NotificationPolicy, get_policy, put_policy, acknowledge, visible_alerts
from surveilx.jobs import jobs
from surveilx.runtime import Runtime, seed_demo
from surveilx.security import (
    cipher,
    create_session,
    current_user,
    hasher,
    identify,
    initialize_admin,
    require,
    revoke,
)
from training.datasets import generate, validate_manifest
from training.import_bundle import import_bundle

runtime = None
login_attempts = {}


@asynccontextmanager
async def lifespan(app):
    global runtime
    Base.metadata.create_all(engine)
    initialize_admin()
    if settings.demo:
        seed_demo()
    runtime = Runtime()
    runtime.restore_models()
    jobs.recover_interrupted()
    evaluation_jobs.recover()
    if settings.start_workers:
        runtime.start()
        adaptation_worker.start()
    yield
    adaptation_worker.stop()
    runtime.stop()


app = FastAPI(title="SurveilX-Edge", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request, call_next):
    if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
        origin = request.headers.get("origin")
        if origin and origin != settings.allowed_origin:
            return Response("Origin not permitted", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


class Login(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=500)


def session_cookie(response, token):
    response.set_cookie(
        "surveilx_session",
        token,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
        max_age=8 * 3600,
        path="/",
    )


@app.post("/api/auth/login")
def login(body: Login, request: Request, response: Response):
    key = request.client.host if request.client else "unknown"
    attempts = [stamp for stamp in login_attempts.get(key, []) if time.monotonic() - stamp < 60]
    if len(attempts) >= 10:
        raise HTTPException(429, "Too many attempts; retry in a minute")
    attempts.append(time.monotonic())
    login_attempts[key] = attempts
    with transaction() as session:
        user = session.scalar(select(User).where(User.username == body.username))
        try:
            if not user or not hasher.verify(user.password_hash, body.password):
                raise HTTPException(401, "Invalid credentials")
        except VerifyMismatchError:
            raise HTTPException(401, "Invalid credentials") from None
        token = create_session(session, user)
        audit(session, user.id, "login")
        session_cookie(response, token)
        return serialize(user)


@app.post("/api/auth/refresh")
def refresh(request: Request, response: Response, user=Depends(current_user)):
    revoke(request.cookies.get("surveilx_session"))
    with transaction() as session:
        token = create_session(session, user)
    session_cookie(response, token)
    return {"refreshed": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user=Depends(current_user)):
    revoke(request.cookies.get("surveilx_session"))
    response.delete_cookie("surveilx_session")
    return {"logged_out": True}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return serialize(user)


class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=2000)
    environment: str = Field(default="custom", min_length=1, max_length=100)
    priority: int = Field(default=1, ge=1, le=5)
    enabled: bool = True
    zones: list[list[float]] = Field(default_factory=list, max_length=20)
    masks: list[list[float]] = Field(default_factory=list, max_length=20)

    @field_validator("zones", "masks")
    @classmethod
    def rectangles(cls, values):
        for box in values:
            if len(box) != 4 or not all(0 <= v <= 1 for v in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Rectangles require normalized x1,y1,x2,y2")
        return values

    @field_validator("source")
    @classmethod
    def source_allowed(cls, value):
        if re.fullmatch(r"demo://\d{1,2}|webcam:\d{1,2}", value):
            return value
        if value.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            return value
        media = (settings.data_dir / "media").resolve()
        path = Path(value).resolve()
        if not path.is_relative_to(media) or not path.is_file():
            raise ValueError("Use a stream URL, webcam:N, demo://N, or existing file under data/media")
        return str(path)


@app.get("/api/cameras")
def cameras(user=Depends(current_user)):
    with transaction() as session:
        values = [serialize(row) for row in session.scalars(select(Camera)).all()]
    live = {row["id"]: row for row in runtime.snapshots()}
    return [row | live.get(row["id"], {"status": "disabled"}) for row in values]


@app.post("/api/cameras", status_code=201)
def add_camera(body: CameraInput, user=Depends(current_user)):
    require(user)
    with transaction() as session:
        if len(session.scalars(select(Camera)).all()) >= settings.max_cameras:
            raise HTTPException(409, "Configured camera limit reached")
        data = body.model_dump(exclude={"source"})
        camera = Camera(**data, source_cipher=cipher().encrypt(body.source.encode()).decode())
        session.add(camera)
        session.flush()
        audit(session, user.id, "camera_created", camera.id)
        result = serialize(camera)
    runtime.sync_cameras()
    return result


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str, user=Depends(current_user)):
    return next((row for row in cameras(user) if row["id"] == camera_id), None) or missing()


def missing():
    raise HTTPException(404, "Record not found")


class CameraPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    priority: int | None = Field(default=None, ge=1, le=5)
    enabled: bool | None = None
    environment: str | None = Field(default=None, min_length=1, max_length=100)
    zones: list[list[float]] | None = Field(default=None, max_length=20)
    masks: list[list[float]] | None = Field(default=None, max_length=20)

    @field_validator("zones", "masks")
    @classmethod
    def rectangles(cls, value):
        if value is not None:
            return CameraInput.rectangles(value)
        return value


@app.patch("/api/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraPatch, user=Depends(current_user)):
    require(user, "operator")
    with transaction() as session:
        camera = session.get(Camera, camera_id) or missing()
        old = serialize(camera)
        for key, value in body.model_dump(exclude_none=True).items():
            setattr(camera, key, value)
        camera.profile_version += 1
        audit(session, user.id, "camera_updated", camera_id, old=old, new=serialize(camera))
    runtime.restart_camera(camera_id)
    return get_camera(camera_id, user)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str, user=Depends(current_user)):
    require(user)
    # Preserve incident foreign keys: deletion is a documented archival operation.
    return update_camera(camera_id, CameraPatch(enabled=False), user)


@app.get("/api/cameras/{camera_id}/health")
def camera_health(camera_id: str, user=Depends(current_user)):
    return get_camera(camera_id, user)


@app.get("/api/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, user=Depends(current_user)):
    capture = runtime.cameras.get(camera_id) or missing()
    with capture.lock:
        if capture.frame is None:
            raise HTTPException(503, "Waiting for camera frame")
        frame = capture.frame.copy()
        outputs = list(capture.outputs)
    h, w = frame.shape[:2]
    for result in outputs:
        x1, y1, x2, y2 = result["box"]
        cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (130, 230, 200), 2)
        cv2.putText(
            frame,
            f"{result['label']} #{result['track_id']}",
            (int(x1 * w), max(15, int(y1 * h) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (130, 230, 200),
            1,
        )
    success, jpeg = cv2.imencode(".jpg", frame)
    if not success:
        raise HTTPException(503, "Frame encoding failed")
    return Response(jpeg.tobytes(), media_type="image/jpeg")


def listing(model, limit=200):
    with transaction() as session:
        query = select(model)
        if hasattr(model, "created"):
            query = query.order_by(model.created.desc())
        elif hasattr(model, "timestamp"):
            query = query.order_by(model.timestamp.desc())
        return [serialize(row) for row in session.scalars(query.limit(limit)).all()]


@app.get("/api/incidents")
def incidents(user=Depends(current_user)):
    return listing(Incident)


@app.get("/api/incidents/{incident_id}")
def incident_detail(incident_id: str, user=Depends(current_user)):
    with transaction() as session:
        return serialize(session.get(Incident, incident_id) or missing())


class IncidentAction(BaseModel):
    state: str
    note: str = Field(default="", max_length=4000)


@app.patch("/api/incidents/{incident_id}")
def change_incident(incident_id: str, body: IncidentAction, user=Depends(current_user)):
    require(user, "operator")
    with transaction() as session:
        incident = session.get(Incident, incident_id) or missing()
        transition(session, incident, body.state, user.id, body.note)
        return serialize(incident)


@app.post("/api/incidents/{incident_id}/{action}")
def incident_action(incident_id: str, action: str, user=Depends(current_user)):
    states = {
        "acknowledge": "ACKNOWLEDGED",
        "resolve": "RESOLVED",
        "dismiss": "FALSE_POSITIVE",
        "escalate": "ESCALATED",
    }
    if action not in states:
        missing()
    return change_incident(incident_id, IncidentAction(state=states[action]), user)


@app.get("/api/incidents/{incident_id}/evidence")
def evidence(incident_id: str, observation: int = Query(default=0, ge=0), user=Depends(current_user)):
    with transaction() as session:
        incident = session.get(Incident, incident_id) or missing()
        if not incident.evidence_key:
            raise HTTPException(410, "Evidence expired or unavailable")
        keys = [incident.evidence_key, *incident.details.get("additional_evidence", [])]
        if observation >= len(keys):
            raise HTTPException(404, "Evidence observation does not exist")
        try:
            payload = read_bundle(keys[observation])
        except FileNotFoundError:
            raise HTTPException(410, "Evidence object is unavailable") from None
        audit(session, user.id, "evidence_accessed", incident.id, observation=observation)
    return Response(
        payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="evidence-{incident_id}-{observation}.zip"'},
    )


@app.get("/api/alerts")
def alerts(user=Depends(current_user)):
    with transaction() as session:
        return visible_alerts(session, user.role)


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, user=Depends(current_user)):
    require(user, "operator")
    with transaction() as session:
        alert = session.get(Alert, alert_id) or missing()
        metadata = session.get(Record, f"notification:{alert.id}")
        if metadata and user.role not in metadata.payload["recipient_roles"]:
            raise HTTPException(403, "Notification is not routed to this role")
        try:
            acknowledge(session, alert, user.id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return serialize(alert)


@app.get("/api/notifications/policy")
def notification_policy(user=Depends(current_user)):
    with transaction() as session:
        return get_policy(session).model_dump()


@app.put("/api/notifications/policy")
def change_notification_policy(body: NotificationPolicy, user=Depends(current_user)):
    require(user)
    with transaction() as session:
        return put_policy(session, body, user.id)


class FeedbackInput(BaseModel):
    incident_id: str
    label: str = Field(pattern="^(true_event|false_positive|ambiguous|missed_event)$")
    notes: str = Field(default="", max_length=4000)


@app.post("/api/feedback")
def feedback(body: FeedbackInput, user=Depends(current_user)):
    require(user, "operator", "researcher")
    with transaction() as session:
        session.get(Incident, body.incident_id) or missing()
        record = Feedback(**body.model_dump(), actor=user.id)
        session.add(record)
        session.flush()
        audit(session, user.id, "feedback_received", record.id)
        return serialize(record)


@app.get("/api/feedback")
def list_feedback(user=Depends(current_user)):
    return listing(Feedback)


@app.post("/api/feedback/{feedback_id}/review")
def review_feedback(feedback_id: str, user=Depends(current_user)):
    require(user, "researcher")
    with transaction() as session:
        row = session.get(Feedback, feedback_id) or missing()
        if row.actor == user.id:
            raise HTTPException(409, "A second reviewer must approve this label")
        row.reviewed = True
        audit(session, user.id, "feedback_reviewed", row.id)
        return serialize(row)


@app.get("/api/incidents/{incident_id}/frames")
def incident_frames(
    incident_id: str, observation: int = Query(default=0, ge=0, le=99), user=Depends(current_user)
):
    require(user, "operator", "researcher")
    with transaction() as session:
        incident = session.get(Incident, incident_id) or missing()
        try:
            _, checksum, names, _ = evidence_frames(incident, observation)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from None
        audit(session, user.id, "annotation_evidence_viewed", incident_id, observation=observation)
        return {
            "incident_id": incident_id,
            "camera_id": incident.camera_id,
            "sha256": checksum,
            "synthetic": bool(incident.details.get("synthetic", False)),
            "frames": [
                {
                    "index": i,
                    "name": name,
                    "url": f"/api/incidents/{incident_id}/frames/{i}?observation={observation}",
                    "review_url": f"/api/incidents/{incident_id}/frames/{i}?observation={observation}&view=review",
                }
                for i, name in enumerate(names)
            ],
        }


@app.get("/api/incidents/{incident_id}/frames/{index}")
def incident_frame(
    incident_id: str,
    index: int,
    observation: int = Query(default=0, ge=0, le=99),
    view: Literal["raw", "review"] = "raw",
    user=Depends(current_user),
):
    require(user, "operator", "researcher")
    with transaction() as session:
        incident = session.get(Incident, incident_id) or missing()
        try:
            _, _, _, frames = evidence_frames(incident, observation, review=view == "review")
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from None
        if not 0 <= index < len(frames):
            missing()
        return Response(frames[index], media_type="image/jpeg")


@app.get("/api/adaptation/annotations")
def annotations(user=Depends(current_user)):
    require(user, "operator", "researcher")
    return records("annotation")


@app.post("/api/adaptation/annotations", status_code=201)
def annotate(body: AnnotationInput, user=Depends(current_user)):
    require(user, "operator", "researcher")
    try:
        return submit_annotation(body, user.id)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from None


class AnnotationReview(BaseModel):
    approve: bool
    notes: str = Field(default="", max_length=4000)


@app.post("/api/adaptation/annotations/{annotation_id}/review")
def approve_annotation(annotation_id: str, body: AnnotationReview, user=Depends(current_user)):
    require(user, "researcher")
    try:
        return review_annotation(annotation_id, user.id, body.approve, body.notes)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@app.post("/api/adaptation/datasets", status_code=201)
def assemble_dataset(body: DatasetBuild, user=Depends(current_user)):
    require(user, "researcher")
    try:
        return build_dataset(body, user.id)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/adaptation/policy")
def get_adaptation_policy(user=Depends(current_user)):
    require(user, "researcher")
    return adaptation_worker.status()


@app.get("/api/adaptation/drift")
def adaptation_drift(user=Depends(current_user)):
    require(user, "researcher")
    return drift_monitor.status()


@app.post("/api/adaptation/policy")
def set_adaptation_policy(body: AdaptationPolicy, user=Depends(current_user)):
    require(user)
    try:
        return adaptation_worker.configure(body, user.id)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/acceptance")
def acceptance_reports(user=Depends(current_user)):
    require(user, "researcher")
    return records("acceptance")


@app.post("/api/models/{model_id}/evaluate", status_code=202)
def evaluate_model(model_id: str, body: AcceptanceInput, user=Depends(current_user)):
    require(user)
    try:
        return {"id": evaluation_jobs.start(model_id, body, user.id), "state": "queued"}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/acceptance/{report_id}/approve")
def approve_acceptance(report_id: str, user=Depends(current_user)):
    require(user)
    try:
        return accept_report(report_id, user.id)
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from None


@app.get("/api/hardware/status")
def hardware(user=Depends(current_user)):
    from surveilx.accelerators import recorded_benchmarks
    from surveilx.hardware_simulation import simulate
    from surveilx.training_runtime import training_runtime_status

    return {
        "hardware": runtime.hardware,
        "power": runtime.power,
        "accelerators": capabilities(),
        "simulation": simulate(),
        "recorded_benchmarks": recorded_benchmarks(settings.data_dir),
        "training_runtime": training_runtime_status(),
    }


@app.get("/api/capabilities")
def task_coverage(user=Depends(current_user)):
    from surveilx.task_profiles import coverage

    datasets = []
    errors = []
    for path in (settings.data_dir / "datasets").glob("*/manifest.json"):
        try:
            datasets.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, OSError):
            errors.append(path.parent.name)
    verified = []
    with transaction() as session:
        for model in session.scalars(select(ModelVersion)):
            if model.manifest.get("acceptance"):
                try:
                    verify_approved_artifact(model, session)
                    verified.append(model.id)
                except (ValueError, OSError):
                    pass
    return {
        "profiles": coverage(listing(ModelVersion), datasets, runtime.active_models.values(), verified),
        "unreadable_datasets": errors,
        "scope": "Declared task coverage; candidate availability does not establish real-world accuracy",
    }


@app.get("/api/controller/status")
def controller(user=Depends(current_user)):
    return runtime.status()


@app.get("/api/generations")
def generation_status(user=Depends(current_user)):
    from surveilx.generations import read_generation, read_queue

    result = []
    seen = set()
    for path in sorted((settings.data_dir / "generations").glob("*/status.json")):
        try:
            result.append(read_generation(path, settings.data_dir))
            seen.add(path.parent.resolve())
        except (ValueError, OSError, TypeError):
            result.append({"generation": path.parent.name, "state": "unreadable", "jobs": []})
    for path in sorted((settings.data_dir / "generations").glob("*/queue.json")):
        if path.parent.resolve() in seen:
            continue
        try:
            result.append(read_queue(path))
        except (ValueError, OSError, TypeError):
            result.append({"generation": path.parent.name, "state": "unreadable", "jobs": []})
    return result


def records(kind):
    with transaction() as session:
        return [
            serialize(row)
            for row in session.scalars(
                select(Record).where(Record.kind == kind).order_by(Record.timestamp.desc()).limit(200)
            ).all()
        ]


@app.get("/api/controller/decisions")
def decisions(user=Depends(current_user)):
    return records("decision")


@app.get("/api/hardware/metrics")
def hardware_metrics(user=Depends(current_user)):
    return records("hardware")


@app.get("/api/audit")
def audit_logs(user=Depends(current_user)):
    require(user, "operator", "researcher", "analyst")
    return listing(Audit)


@app.get("/api/models")
def models(user=Depends(current_user)):
    return listing(ModelVersion)


@app.get("/api/models/{model_id}")
def get_model(model_id: str, user=Depends(current_user)):
    with transaction() as session:
        return serialize(session.get(ModelVersion, model_id) or missing())


class DeployRequest(BaseModel):
    stage: str = Field(default="canary", pattern="^(canary|production)$")
    slot: str = Field(default="default", pattern="^[a-zA-Z0-9_-]{1,48}$")


@app.get("/api/cameras/{camera_id}/policies")
def get_site_policy(camera_id: str, user=Depends(current_user)):
    with transaction() as session:
        session.get(Camera, camera_id) or missing()
        row = session.get(Record, f"site-policy:{camera_id}")
        return row.payload if row else {"rules": []}


@app.put("/api/cameras/{camera_id}/policies")
def set_site_policy(camera_id: str, body: SitePolicy, user=Depends(current_user)):
    require(user, "operator")
    with transaction() as session:
        session.get(Camera, camera_id) or missing()
        row = session.get(Record, f"site-policy:{camera_id}")
        if row:
            row.payload, row.timestamp = body.model_dump(), time.time()
        else:
            session.add(Record(id=f"site-policy:{camera_id}", kind="site_policy", payload=body.model_dump()))
        audit(session, user.id, "site_policy_updated", camera_id, rules=[rule.id for rule in body.rules])
    runtime.restart_camera(camera_id)
    return body.model_dump()


@app.post("/api/models/{model_id}/deploy")
def deploy_model(model_id: str, body: DeployRequest, user=Depends(current_user)):
    require(user)
    with runtime.deployment_lock:
        with transaction() as session:
            model = session.get(ModelVersion, model_id) or missing()
            task = runtime.model_task(model.manifest)
            key = task if body.slot == "default" else f"{task}:{body.slot}"
            deployments = list(session.scalars(select(Record).where(Record.kind == "deployment")))
            if any(
                row.id != f"deployment:{key}" and row.payload.get("active_id") == model.id
                for row in deployments
            ):
                raise HTTPException(409, "A model can be active in only one deployment slot")
            occupied = [
                row for row in deployments if row.payload.get("active_id") and row.payload.get("task") == task
            ]
            state = session.get(Record, f"deployment:{key}")
            if (not state or not state.payload.get("active_id")) and len(occupied) >= 8:
                raise HTTPException(409, "At most eight active slots per task are supported")
            previous = state.payload if state else {}
            active_id = previous.get("active_id")
            if body.stage == "production" and (
                model.manifest.get("synthetic") or not model.manifest.get("deployment_eligible")
            ):
                raise HTTPException(
                    409, "Candidate has not passed real-domain acceptance; production deployment blocked"
                )
            if body.stage == "production" and (model.stage != "canary" or active_id != model.id):
                raise HTTPException(409, "Production promotion requires this model to be the active canary")
            if body.stage == "production":
                try:
                    verify_approved_artifact(model, session)
                except (ValueError, OSError) as exc:
                    raise HTTPException(409, str(exc)) from None
            try:
                expert = runtime.prepare_model(model)
            except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
                raise HTTPException(409, f"Artifact cannot be activated: {exc}") from None
            if active_id != model.id:
                old = session.get(ModelVersion, active_id) if active_id else None
                previous = {"previous_id": active_id, "previous_stage": old.stage if old else None}
                if old:
                    old.stage = "superseded"
            protected = {row.payload.get("active_id") for row in deployments if row.id != f"deployment:{key}"}
            # Reconcile stale stage flags while retaining unrelated active specialists.
            for other in session.scalars(
                select(ModelVersion).where(ModelVersion.stage.in_(["canary", "production"]))
            ):
                if (
                    other.id != model.id
                    and other.id not in protected
                    and runtime.model_task(other.manifest) == task
                ):
                    other.stage = "superseded"
            payload = {
                **previous,
                "active_id": model.id,
                "task": task,
                "slot": body.slot,
                "stage": body.stage,
            }
            if state:
                state.payload, state.timestamp = payload, time.time()
            else:
                session.add(Record(id=f"deployment:{key}", kind="deployment", payload=payload))
            model.stage = body.stage
            audit(session, user.id, "model_activated", model.id, stage=model.stage, task=task)
            result = serialize(model)
        # Publish only after the database commit. Restart reads the same durable record.
        runtime.install_model(key, expert, model.id)
        return result


@app.post("/api/models/{model_id}/rollback")
def rollback_model(model_id: str, user=Depends(current_user)):
    require(user)
    with runtime.deployment_lock:
        with transaction() as session:
            model = session.get(ModelVersion, model_id) or missing()
            task = runtime.model_task(model.manifest)
            state = next(
                (
                    row
                    for row in session.scalars(select(Record).where(Record.kind == "deployment"))
                    if row.payload.get("active_id") == model.id
                ),
                None,
            )
            if not state or state.payload.get("active_id") != model.id:
                raise HTTPException(409, "Only the currently active model can be rolled back")
            previous_id = state.payload.get("previous_id")
            previous = session.get(ModelVersion, previous_id) if previous_id else None
            if previous and any(
                row.id != state.id and row.payload.get("active_id") == previous_id
                for row in session.scalars(select(Record).where(Record.kind == "deployment"))
            ):
                raise HTTPException(
                    409, "Previous model is active in another slot; current deployment retained"
                )
            if previous_id and not previous:
                raise HTTPException(409, "Previous model is unavailable; current deployment retained")
            try:
                if previous and state.payload.get("previous_stage") == "production":
                    verify_approved_artifact(previous, session)
                expert = runtime.prepare_model(previous) if previous else None
            except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
                raise HTTPException(409, f"Rollback artifact cannot be activated: {exc}") from None
            model.stage = "rollback"
            previous_stage = state.payload.get("previous_stage") or "canary"
            if previous:
                previous.stage = previous_stage
            state.payload = {
                "active_id": previous_id,
                "previous_id": None,
                "task": task,
                "slot": state.payload.get("slot", "default"),
                "stage": previous_stage if previous else None,
            }
            state.timestamp = time.time()
            audit(session, user.id, "model_rolled_back", model_id, restored_id=previous_id, task=task)
        runtime.install_model(state.id.removeprefix("deployment:"), expert, previous_id)
    return {"stage": "rollback", "restored_model_id": previous_id}


class GeneratedDataset(BaseModel):
    name: str = Field(pattern="^[a-zA-Z0-9_-]{1,64}$")
    count: int = Field(default=160, ge=80, le=2000)
    seed: int = Field(default=42, ge=0)


@app.get("/api/datasets")
def datasets(user=Depends(current_user)):
    result = []
    for path in (settings.data_dir / "datasets").glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        result.append(
            {
                "name": path.parent.name,
                "task": manifest.get("task", "event"),
                "synthetic": manifest.get("synthetic", False),
                "samples": len(manifest["samples"]),
                "domain": manifest["domain"],
                "classes": manifest["classes"],
            }
        )
    return result


@app.post("/api/datasets/generate")
def generate_dataset(body: GeneratedDataset, user=Depends(current_user)):
    require(user, "researcher")
    try:
        generate(settings.data_dir / "datasets" / body.name, body.count, body.seed)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    with transaction() as session:
        audit(session, user.id, "synthetic_dataset_generated", body.name, count=body.count)
    return {"name": body.name}


@app.post("/api/datasets/{name}/validate")
def validate_dataset(name: str, user=Depends(current_user)):
    require(user, "researcher")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise HTTPException(422, "Invalid dataset name")
    try:
        _, counts = validate_manifest(settings.data_dir / "datasets" / name / "manifest.json")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    return {"valid": True, "counts": counts}


@app.post("/api/datasets/{name}/import")
async def import_dataset(name: str, request: Request, user=Depends(current_user)):
    require(user, "researcher")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise HTTPException(422, "Invalid dataset name")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > 100 * 1024**2:
            raise HTTPException(413, "Compressed import limit is 100 MB")
    try:
        result = await asyncio.to_thread(import_bundle, payload, settings.data_dir / "datasets" / name)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    with transaction() as session:
        audit(session, user.id, "dataset_imported", name)
    return result


class TrainRequest(BaseModel):
    dataset: str = Field(pattern="^[a-zA-Z0-9_-]{1,64}$")
    epochs: int = Field(default=5, ge=1, le=500)
    architecture: str = Field(default="auto", pattern="^(auto|scratch|yolo_rai)$")


@app.post("/api/experiments")
def train_request(body: TrainRequest, user=Depends(current_user)):
    require(user, "researcher")
    validate_dataset(body.dataset, user)
    try:
        job_id = jobs.start(body.dataset, body.epochs, user.id, body.architecture)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"id": job_id, "state": "queued"}


@app.get("/api/experiments")
def experiments(user=Depends(current_user)):
    return records("experiment")


class UserInput(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=12, max_length=500)
    role: str = Field(pattern="^(admin|operator|researcher|analyst|viewer)$")


@app.get("/api/users")
def users(user=Depends(current_user)):
    require(user)
    return listing(User)


@app.post("/api/users")
def add_user(body: UserInput, user=Depends(current_user)):
    require(user)
    with transaction() as session:
        if session.scalar(select(User).where(User.username == body.username)):
            raise HTTPException(409, "Username exists")
        created = User(username=body.username, password_hash=hasher.hash(body.password), role=body.role)
        session.add(created)
        session.flush()
        audit(session, user.id, "user_created", created.id, role=created.role)
        return serialize(created)


@app.get("/api/settings")
def public_settings(user=Depends(current_user)):
    return {
        key: getattr(settings, key)
        for key in [
            "coverage_seconds",
            "confirmation_seconds",
            "cooldown_seconds",
            "retention_days",
            "max_cameras",
            "epoch_seconds",
        ]
    }


@app.get("/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok", "scheduler": runtime.heartbeat if runtime else None}
    except Exception:
        raise HTTPException(503, "Database unavailable") from None


@app.websocket("/live/{channel}")
async def live(socket: WebSocket, channel: str):
    if socket.headers.get("origin") != settings.allowed_origin or channel not in {
        "cameras",
        "incidents",
        "alerts",
        "system",
        "metrics",
    }:
        await socket.close(code=1008)
        return
    try:
        identify(socket.cookies.get("surveilx_session"))
    except HTTPException:
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        while True:
            user = await asyncio.to_thread(identify, socket.cookies.get("surveilx_session"))
            if channel == "cameras":
                payload = runtime.snapshots()
            elif channel in {"system", "metrics"}:
                payload = runtime.status()
            elif channel == "alerts":
                payload = await asyncio.to_thread(alerts, user)
            else:
                payload = await asyncio.to_thread(listing, Incident)
            await socket.send_json({"channel": channel, "timestamp": time.time(), "data": payload})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
    except (RuntimeError, HTTPException):
        try:
            await socket.close(code=1008)
        except RuntimeError:
            pass


frontend = Path("frontend/dist")
if frontend.exists():
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    if path.startswith(("api/", "live/")):
        missing()
    if (frontend / "index.html").exists():
        return FileResponse(frontend / "index.html")
    return {"message": "Build frontend with npm run build in frontend; API docs at /docs"}
