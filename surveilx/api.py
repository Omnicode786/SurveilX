import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, text

from surveilx.accelerators import capabilities
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
from surveilx.incidents import transition
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
    if settings.start_workers:
        runtime.start()
    yield
    if settings.start_workers:
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
    environment: str = Field(
        default="custom", pattern="^(parking|office|warehouse|industrial|retail|traffic|building|custom)$"
    )
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
    environment: str | None = Field(
        default=None, pattern="^(parking|office|warehouse|industrial|retail|traffic|building|custom)$"
    )
    zones: list[list[float]] | None = None
    masks: list[list[float]] | None = None

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
def evidence(incident_id: str, user=Depends(current_user)):
    with transaction() as session:
        incident = session.get(Incident, incident_id) or missing()
        if not incident.evidence_key:
            raise HTTPException(410, "Evidence expired or unavailable")
        payload = read_bundle(incident.evidence_key)
        audit(session, user.id, "evidence_accessed", incident.id)
    return Response(
        payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="evidence-{incident_id}.zip"'},
    )


@app.get("/api/alerts")
def alerts(user=Depends(current_user)):
    return listing(Alert)


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, user=Depends(current_user)):
    require(user, "operator")
    with transaction() as session:
        alert = session.get(Alert, alert_id) or missing()
        alert.acknowledged_by, alert.status = user.id, "acknowledged"
        audit(session, user.id, "alert_acknowledged", alert_id)
        return serialize(alert)


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


@app.get("/api/hardware/status")
def hardware(user=Depends(current_user)):
    return {"hardware": runtime.hardware, "power": runtime.power, "accelerators": capabilities()}


@app.get("/api/controller/status")
def controller(user=Depends(current_user)):
    return runtime.status()


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


@app.post("/api/models/{model_id}/deploy")
def deploy_model(model_id: str, body: DeployRequest, user=Depends(current_user)):
    require(user)
    with transaction() as session:
        model = session.get(ModelVersion, model_id) or missing()
        if body.stage == "production" and (
            model.manifest.get("synthetic") or not model.manifest.get("deployment_eligible")
        ):
            raise HTTPException(
                409, "Candidate has not passed real-domain acceptance; production deployment blocked"
            )
        if body.stage == "production" and model.stage != "canary":
            raise HTTPException(409, "Production promotion requires a canary first")
        version = model.version
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
            raise HTTPException(422, "Invalid artifact version")
        try:
            runtime.activate_sva(settings.data_dir / "runs" / version)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from None
        model.stage = body.stage
        audit(session, user.id, "model_activated", model.id, stage=model.stage)
        return serialize(model)


@app.post("/api/models/{model_id}/rollback")
def rollback_model(model_id: str, user=Depends(current_user)):
    require(user)
    with transaction() as session:
        model = session.get(ModelVersion, model_id) or missing()
        if model.stage not in {"production", "canary"}:
            raise HTTPException(409, "Model is not deployed")
        model.stage = "rollback"
        runtime.rollback_sva()
        audit(session, user.id, "model_rolled_back", model_id)
    return {"stage": "rollback"}


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


@app.post("/api/experiments")
def train_request(body: TrainRequest, user=Depends(current_user)):
    require(user, "researcher")
    validate_dataset(body.dataset, user)
    try:
        job_id = jobs.start(body.dataset, body.epochs, user.id)
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
            await asyncio.to_thread(identify, socket.cookies.get("surveilx_session"))
            if channel == "cameras":
                payload = runtime.snapshots()
            elif channel in {"system", "metrics"}:
                payload = runtime.status()
            else:
                payload = await asyncio.to_thread(listing, Incident if channel == "incidents" else Alert)
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
