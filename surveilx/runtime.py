import json
import logging
import re
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
from sqlalchemy import select

from surveilx.config import settings
from surveilx.controller import Candidate, Scheduler
from surveilx.database import Camera, Incident, ModelVersion, Record, audit, transaction
from surveilx.drift import drift_monitor
from surveilx.evidence import delete_bundle, save_bundle
from surveilx.hardware import PowerGovernor, probe
from surveilx.incidents import create_incident
from surveilx.notifications import process_due
from surveilx.security import cipher
from surveilx.policies import PolicyEngine
from surveilx.vision import HOGDetector, SyntheticDetector, Tracker, YOLODetector, generated_frame, redact

logger = logging.getLogger("surveilx")


class Capture:
    def __init__(self, camera):
        self.id, self.name = camera.id, camera.name
        self.source = cipher().decrypt(camera.source_cipher.encode()).decode()
        self.masks, self.zones = camera.masks, camera.zones
        self.priority, self.environment = camera.priority, camera.environment
        self.synthetic = self.source.startswith("demo://")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.sequence = 0
        self.consumed = 0
        self.dropped = 0
        self.reconnects = 0
        self.status = "connecting"
        self.fps = 5
        self.last_service = time.monotonic()
        self.latency_ms = 50.0
        self.outputs = []
        self.detector = "pending"
        self.calibration = "uncalibrated"
        self.tracker = Tracker()
        self.buffer = deque(maxlen=30)
        self.candidate_since = None
        self.last_incident = 0.0
        self.last_detection_log = 0.0
        self.last_detection_signature = ()
        self.previous_gray = None
        self.entity_history = deque(maxlen=64)
        self.expert_states = {}
        self.expert_last_seen = {}
        self.expert_cursor = 0
        self.expert_key = None
        self.policy_engine = PolicyEngine({"rules": []})
        # Model contracts cap clips at 64 observations; retaining more wastes edge memory.
        self.scene_history = deque(maxlen=64)
        self.scene_times = deque(maxlen=64)
        self.secondary_result = None
        self.motion = 0.0
        self.brightness = 0.0
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"capture-{self.id}")

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=4)

    def run(self):
        connection = None
        tick = 0
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                if self.synthetic:
                    frame = generated_frame(int(self.source.split("//")[1]), tick / self.fps)
                else:
                    if connection is None:
                        source = int(self.source[7:]) if self.source.startswith("webcam:") else self.source
                        connection = cv2.VideoCapture()
                        if isinstance(source, str) and source.startswith(
                            ("rtsp://", "rtsps://", "http://", "https://")
                        ):
                            connection.open(
                                source,
                                cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000],
                            )
                        else:
                            connection.open(source)
                        connection.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame = connection.read()
                    if not ok:
                        raise RuntimeError("Source unavailable or end of file")
                frame = redact(frame, self.masks)
                # Bound retained frames independently of source resolution.
                height, width = frame.shape[:2]
                scale = min(1.0, 640 / max(height, width))
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
                with self.lock:
                    if self.sequence > self.consumed:
                        self.dropped += 1
                    self.frame = frame
                    self.timestamp = time.time()
                    self.sequence += 1
                    self.status = "online"
                    self.brightness = float(gray.mean() / 255)
                    self.motion = (
                        float(np.mean(cv2.absdiff(gray, self.previous_gray) > 20))
                        if self.previous_gray is not None
                        else 0
                    )
                    self.previous_gray = gray
                    self.buffer.append((self.timestamp, frame))
                tick += 1
            except Exception:
                self.status = "reconnecting"
                self.reconnects += 1
                if connection is not None:
                    connection.release()
                    connection = None
                self.stop_event.wait(min(10, 1 + self.reconnects / 2))
            self.stop_event.wait(max(0, 1 / self.fps - (time.monotonic() - start)))
        if connection is not None:
            connection.release()
        self.status = "stopped"

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "synthetic": self.synthetic,
                "environment": self.environment,
                "priority": self.priority,
                "capture_fps_target": self.fps,
                "frames": self.sequence,
                "dropped_frames": self.dropped,
                "reconnects": self.reconnects,
                "latency_ms": round(self.latency_ms, 2),
                "model": self.detector,
                "service_age_seconds": round(time.monotonic() - self.last_service, 2),
                "last_frame": self.timestamp,
                "detections": self.outputs,
                "motion": self.motion,
                "brightness": self.brightness,
                "calibration": self.calibration,
                "secondary_evidence": self.secondary_result,
                "risk": "review" if self.candidate_since else "no active zone observation",
            }


class Runtime:
    def __init__(self):
        self.cameras = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.governor = PowerGovernor()
        self.scheduler = Scheduler()
        self.hardware = probe().json()
        self.power = {"level": "economy", "resolution": 320, "budget_ms": 200, "capture_fps": 5}
        self.decision = {}
        self.heartbeat = 0.0
        self.error = None
        self.latencies = deque(maxlen=1000)
        self.baseline = HOGDetector()
        self.synthetic_detector = SyntheticDetector()
        self.yolo = None
        self.sva = None
        self.sva_lock = threading.Lock()
        self.deployment_lock = threading.RLock()
        self.learned_detector = None
        self.active_models = {"event": None, "detection": None}
        self.experts = {}
        self.started = False
        if settings.detector_path:
            try:
                self.yolo = YOLODetector(settings.detector_path)
            except Exception as exc:
                self.error = f"Detector load failed: {type(exc).__name__}; HOG fallback active"
        self.thread = threading.Thread(target=self.run, daemon=True, name="global-inference")

    @staticmethod
    def model_task(manifest):
        return "detection" if manifest.get("task") == "detection" else "event"

    def prepare_model(self, model):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", model.version):
            raise ValueError("Invalid artifact version")
        directory = settings.data_dir / "runs" / model.version
        if self.model_task(model.manifest) == "detection":
            from surveilx.detector_expert import ScratchDetector, AdaptedYOLODetector

            if model.manifest.get("architecture") in {"yolo_rai", "yolo_baseline", "yolo_pretrained"}:
                return AdaptedYOLODetector(directory)
            return ScratchDetector(directory)
        from surveilx.expert import SVAExpert

        return SVAExpert(directory)

    def install_model(self, task, expert, model_id):
        with self.sva_lock:
            if task == "detection":
                self.learned_detector = expert
            elif task == "event":
                self.sva = expert
            if expert is None:
                self.experts.pop(task, None)
            else:
                self.experts[task] = expert
            self.active_models[task] = model_id

    def restore_models(self):
        """Restore only committed deployment state; never guess between stale stage flags."""
        with transaction() as session:
            tasks = [
                state.id.removeprefix("deployment:")
                for state in session.scalars(select(Record).where(Record.kind == "deployment"))
            ]
        for task in tasks:
            try:
                with transaction() as session:
                    state = session.get(Record, f"deployment:{task}")
                    if not state or not state.payload.get("active_id"):
                        continue
                    model = session.get(ModelVersion, state.payload["active_id"])
                    if model is None or model.stage not in {"canary", "production"}:
                        raise ValueError("Deployment registry is inconsistent")
                    if task.split(":")[0] != self.model_task(model.manifest):
                        raise ValueError("Deployment task differs from model task")
                    if model.stage == "production":
                        from surveilx.acceptance import verify_approved_artifact

                        verify_approved_artifact(model, session)
                    expert = self.prepare_model(model)
                    self.install_model(task, expert, model.id)
            except Exception as exc:
                self.error = (
                    f"{task.title()} deployment restore failed: {type(exc).__name__}; baseline retained"
                )
                logger.error(self.error)

    def start(self):
        self.started = True
        self.sync_cameras()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        for capture in list(self.cameras.values()):
            capture.stop()
        self.started = False

    def sync_cameras(self):
        with transaction() as session:
            cameras = session.scalars(select(Camera).where(Camera.enabled.is_(True))).all()
        with self.lock:
            ids = {camera.id for camera in cameras}
            for key in list(self.cameras):
                if key not in ids:
                    self.cameras.pop(key).stop()
            for camera in cameras:
                if camera.id not in self.cameras:
                    self.cameras[camera.id] = Capture(camera)
                    with transaction() as session:
                        policy = session.get(Record, f"site-policy:{camera.id}")
                        if policy:
                            self.cameras[camera.id].policy_engine = PolicyEngine(policy.payload)
                if self.started and not self.cameras[camera.id].thread.is_alive():
                    self.cameras[camera.id].start()

    def restart_camera(self, camera_id):
        with self.lock:
            old = self.cameras.pop(camera_id, None)
            if old:
                old.stop()
        self.sync_cameras()

    def snapshots(self):
        with self.lock:
            return [camera.snapshot() for camera in self.cameras.values()]

    def status(self):
        return {
            "hardware": self.hardware,
            "power": self.power,
            "decision": self.decision,
            "heartbeat": self.heartbeat,
            "error": self.error,
            "active_models": dict(self.active_models),
            "mean_latency_ms": float(np.mean(self.latencies)) if self.latencies else None,
            "p95_latency_ms": float(np.percentile(self.latencies, 95)) if self.latencies else None,
            "cameras": self.snapshots(),
            "database": "sqlite-local" if settings.database_url.startswith("sqlite") else "postgresql",
        }

    def persist_event(self, payload):
        try:
            with transaction() as session:
                create_incident(session, **payload)
        except Exception:
            spool = settings.data_dir / "spool"
            spool.mkdir(exist_ok=True)
            (spool / f"{uuid.uuid4()}.event").write_bytes(cipher().encrypt(json.dumps(payload).encode()))
            self.error = "Database write failed; encrypted event buffered for retry"

    def flush_spool(self):
        for path in (settings.data_dir / "spool").glob("*.event"):
            try:
                payload = json.loads(cipher().decrypt(path.read_bytes()))
                with transaction() as session:
                    # create_incident also checks grouped evidence to make replay idempotent.
                    create_incident(session, **payload)
                path.unlink()
            except Exception as exc:
                self.error = f"Buffered event retry failed: {type(exc).__name__}"
                logger.error(json.dumps({"event": "spool_retry_failed", "file": path.name}))

    def retention(self):
        cutoff = time.time() - settings.retention_days * 86400
        with transaction() as session:
            expired = session.scalars(
                select(Incident).where(Incident.created < cutoff, Incident.evidence_key.is_not(None))
            ).all()
            for incident in expired:
                delete_bundle(incident.evidence_key)
                for key in incident.details.get("additional_evidence", []):
                    delete_bundle(key)
                incident.details = {**incident.details, "additional_evidence": []}
                incident.evidence_key = None
                audit(session, "retention", "evidence_expired", incident.id)

    def infer(self, capture):
        with capture.lock:
            if capture.frame is None or capture.sequence <= capture.consumed:
                return
            frame = capture.frame.copy()
            captured_at = capture.timestamp
            capture.consumed = capture.sequence
        started = time.perf_counter()
        with self.sva_lock:
            learned = self.learned_detector
            specialists = [
                (key, value)
                for key, value in sorted(self.experts.items())
                if key.startswith("detection:")
                and capture.synthetic == bool(value.manifest.get("synthetic", False))
                and capture.environment == value.manifest.get("domain")
            ]
        detector = self.synthetic_detector if capture.synthetic else (self.yolo or self.baseline)
        selected_key = "baseline"
        if (
            learned
            and capture.synthetic == bool(learned.manifest.get("synthetic", False))
            and (capture.synthetic or capture.environment == learned.manifest.get("domain"))
        ):
            detector = learned
            selected_key = "detection"
        if specialists:
            choices = ([(selected_key, detector)] if selected_key == "detection" else []) + specialists
            selected_key, detector = choices[capture.expert_cursor % len(choices)]
            capture.expert_cursor += 1
        # Keep tracks, temporal history and occupancy persistence isolated by artifact.
        key = (selected_key, getattr(detector, "version", detector.name))
        if capture.expert_key != key:
            fields = ("tracker", "entity_history", "candidate_since", "last_incident", "secondary_result")
            if capture.expert_key is not None:
                capture.expert_states[capture.expert_key] = {
                    field: getattr(capture, field) for field in fields
                }
            state = capture.expert_states.get(key, {})
            if capture.expert_key is not None:
                capture.tracker = state.get("tracker", Tracker())
                capture.entity_history = state.get("entity_history", deque(maxlen=64))
                capture.candidate_since = state.get("candidate_since")
                capture.last_incident = state.get("last_incident", 0.0)
                capture.secondary_result = state.get("secondary_result")
            capture.expert_key = key
            # Bound histories after repeated deployments, including removed artifacts.
            while len(capture.expert_states) > 8:
                capture.expert_states.pop(next(iter(capture.expert_states)))
        last_seen = capture.expert_last_seen.get(key)
        if last_seen is not None and time.monotonic() - last_seen > settings.coverage_seconds:
            capture.candidate_since = None
            capture.entity_history.clear()
        capture.expert_last_seen = {
            k: v for k, v in capture.expert_last_seen.items() if k in capture.expert_states or k == key
        }
        capture.expert_last_seen[key] = time.monotonic()
        try:
            detections = detector.infer(frame, self.power["resolution"])
        except Exception as exc:
            self.error = f"Expert failed: {type(exc).__name__}; baseline fallback"
            detector = self.synthetic_detector if capture.synthetic else self.baseline
            detections = detector.infer(frame, 320)
            capture.tracker = Tracker()
            capture.entity_history.clear()
            capture.candidate_since = None
            capture.expert_key = ("fallback", detector.name)
        detections = capture.tracker.update(detections)
        outputs = [d.json() for d in detections]
        compressed, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        history_frame = encoded.tobytes() if compressed else frame
        # Both histories share the immutable compressed object. Named detection histories may
        # retain independent deques, so compression keeps multi-specialist memory bounded.
        capture.entity_history.append((history_frame, outputs))
        capture.scene_history.append((history_frame, []))
        capture.scene_times.append(captured_at)
        # Secondary computation only on active candidates and compatible, explicitly activated experts.
        with self.sva_lock:
            expert = self.sva
            event_slot = "event:default"
            events = [
                (name, value)
                for name, value in sorted(self.experts.items())
                if name.startswith("event:")
                and capture.environment == value.manifest.get("domain")
                and capture.synthetic == bool(value.manifest.get("synthetic", False))
            ]
            if events:
                event_slot, expert = events[capture.expert_cursor % len(events)]
        if not specialists:
            capture.expert_cursor += 1
        capture.secondary_result = None
        scene_expert = bool(expert and getattr(expert, "scene", False))
        if expert and (capture.candidate_since is not None or scene_expert):
            try:
                capture.secondary_result = expert.infer(
                    list(capture.scene_history if scene_expert else capture.entity_history),
                    capture.synthetic,
                    capture.environment,
                    **({"timestamps": list(capture.scene_times)} if scene_expert else {}),
                )
            except Exception as exc:
                capture.secondary_result = {"decision": "abstain", "reason": "Secondary expert failed"}
                self.error = f"Secondary expert failed: {type(exc).__name__}; primary detections retained"
        elapsed = (time.perf_counter() - started) * 1000
        self.latencies.append(elapsed)
        now = time.monotonic()
        if now - capture.last_service > settings.coverage_seconds:
            capture.candidate_since = None
        with capture.lock:
            capture.outputs = outputs
            capture.detector = detector.name
            capture.calibration = (
                "fitted_detection_correctness" if getattr(detector, "calibrated", False) else "uncalibrated"
            )
            capture.latency_ms = elapsed
            capture.last_service = now
        occupied = False
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            for zone in capture.zones:
                left, top, right, bottom = zone
                occupied |= left <= center[0] <= right and top <= center[1] <= bottom
        evidence_saved = False
        if not occupied:
            capture.candidate_since = None
            if not scene_expert:
                capture.secondary_result = None
        elif capture.candidate_since is None:
            capture.candidate_since = now
        elif (
            now - capture.candidate_since >= settings.confirmation_seconds
            and now - capture.last_incident >= settings.cooldown_seconds
        ):
            details = {
                "detections": capture.outputs,
                "model": detector.name,
                "model_version": getattr(detector, "version", None),
                "expert_slot": selected_key,
                "calibration": capture.calibration,
                "decision": self.decision,
                "power": self.power,
                "synthetic": capture.synthetic,
                "captured_at": captured_at,
                "observed_labels": sorted({item["label"] for item in outputs}),
                "secondary_evidence": capture.secondary_result,
                "interpretation": "Persistent configured-zone occupancy; requires human verification",
            }
            with capture.lock:
                frames = list(capture.buffer)
            key = save_bundle(frames, details)
            self.persist_event(
                {
                    "camera_id": capture.id,
                    "evidence_key": key,
                    "details": details,
                    "synthetic": capture.synthetic,
                }
            )
            capture.last_incident = now
            evidence_saved = True
        for observation in capture.policy_engine.evaluate(outputs, now, capture.expert_key):
            details = {
                "detections": outputs,
                "model": detector.name,
                "model_version": getattr(detector, "version", None),
                "expert_slot": selected_key,
                "calibration": capture.calibration,
                "synthetic": capture.synthetic,
                "captured_at": captured_at,
                "policy_observation": observation,
                "observed_labels": sorted({item["label"] for item in outputs}),
                "interpretation": observation["interpretation"],
            }
            with capture.lock:
                frames = list(capture.buffer)
            evidence = save_bundle(frames, details)
            self.persist_event(
                {
                    "camera_id": capture.id,
                    "evidence_key": evidence,
                    "details": details,
                    "synthetic": capture.synthetic,
                }
            )
            evidence_saved = True
        event_key = (event_slot, getattr(expert, "version", None))
        for observation in capture.policy_engine.evaluate_event(capture.secondary_result, now, event_key):
            details = {
                "detections": outputs,
                "model": getattr(expert, "version", "event-expert"),
                "model_version": getattr(expert, "version", None),
                "expert_slot": event_slot,
                "calibration": "fitted_event_classification",
                "synthetic": capture.synthetic,
                "captured_at": captured_at,
                "policy_observation": observation,
                "observed_labels": [observation["label"]],
                "secondary_evidence": capture.secondary_result,
                "interpretation": observation["interpretation"],
            }
            with capture.lock:
                frames = list(capture.buffer)
            evidence = save_bundle(frames, details)
            self.persist_event(
                {
                    "camera_id": capture.id,
                    "evidence_key": evidence,
                    "details": details,
                    "synthetic": capture.synthetic,
                }
            )
            evidence_saved = True
        signature = tuple(sorted({item["label"] for item in outputs}))
        if evidence_saved and signature:
            capture.last_detection_log = now
        if (
            signature
            and not evidence_saved
            and (
                signature != capture.last_detection_signature
                or now - capture.last_detection_log >= max(1.0, settings.detection_log_seconds)
            )
        ):
            details = {
                "event_type": "detection_sequence",
                "detections": outputs,
                "model": detector.name,
                "model_version": getattr(detector, "version", None),
                "expert_slot": selected_key,
                "calibration": capture.calibration,
                "synthetic": capture.synthetic,
                "captured_at": captured_at,
                "observed_labels": list(signature),
                "secondary_evidence": capture.secondary_result,
                "interpretation": "Detected objects recorded for later review; no threat conclusion is implied",
            }
            with capture.lock:
                frames = list(capture.buffer)
            evidence = save_bundle(frames, details)
            self.persist_event(
                {
                    "camera_id": capture.id,
                    "evidence_key": evidence,
                    "details": details,
                    "synthetic": capture.synthetic,
                }
            )
            capture.last_detection_log = now
        capture.last_detection_signature = signature
        with transaction() as session:
            session.add(
                Record(
                    kind="decision",
                    payload={
                        "timestamp": time.time(),
                        "captured_at": captured_at,
                        "camera_id": capture.id,
                        "action": detector.name,
                        "model_version": getattr(detector, "version", None),
                        "expert_slot": selected_key,
                        "power_level": self.power.get("level"),
                        "resolution": getattr(detector, "manifest", {})
                        .get("config", {})
                        .get("image_size", self.power["resolution"]),
                        "latency_ms": elapsed,
                        "state": {"motion": capture.motion, "brightness": capture.brightness},
                        "result": capture.outputs,
                        "secondary": capture.secondary_result,
                        "calibration": capture.calibration,
                    },
                )
            )

    def run(self):
        epoch = 0
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                hardware = probe()
                self.hardware = hardware.json()
                recent_latency = float(np.percentile(list(self.latencies)[-20:], 95)) if self.latencies else 0
                self.power = self.governor.update(hardware, recent_latency)
                with self.lock:
                    captures = list(self.cameras.values())
                candidates = []
                for capture in captures:
                    capture.fps = self.power["capture_fps"]
                    age = time.monotonic() - capture.last_service
                    # Floor is cold-start reservation, not a fabricated benchmark.
                    candidates.append(
                        Candidate(
                            capture.id,
                            capture.priority,
                            age,
                            max(10, capture.latency_ms * 1.25),
                            capture.status == "online" and capture.sequence > capture.consumed,
                        )
                    )
                self.decision = self.scheduler.allocate(
                    candidates, self.power["budget_ms"], settings.coverage_seconds, settings.epoch_seconds
                )
                for camera_id in self.decision["selected"]:
                    capture = next((c for c in captures if c.id == camera_id), None)
                    if capture:
                        try:
                            self.infer(capture)
                        except Exception as exc:
                            self.error = f"Inference/persistence failure: {type(exc).__name__}"
                if epoch % 10 == 0:
                    self.flush_spool()
                    with transaction() as session:
                        session.add(Record(kind="hardware", payload=self.hardware | {"power": self.power}))
                        process_due(session)
                    self.sync_cameras()
                if epoch % 60 == 0:
                    drift_monitor.evaluate()
                if epoch % 600 == 0:
                    self.retention()
                self.heartbeat = time.time()
                epoch += 1
            except Exception as exc:
                self.error = f"Runtime degraded: {type(exc).__name__}"
                logger.error(json.dumps({"event": "runtime_error", "type": type(exc).__name__}))
            self.stop_event.wait(max(0.05, settings.epoch_seconds - (time.monotonic() - start)))


def seed_demo():
    with transaction() as session:
        if session.scalar(select(Camera).limit(1)):
            return
        for index, environment in enumerate(["parking", "office", "warehouse", "retail"]):
            session.add(
                Camera(
                    name=f"Test {index + 1:02d} · {environment.title()}",
                    source_cipher=cipher().encrypt(f"demo://{index}".encode()).decode(),
                    environment=environment,
                    zones=[[0.6, 0.2, 0.96, 0.9]],
                    priority=index % 3 + 1,
                )
            )
