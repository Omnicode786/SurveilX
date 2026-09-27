import json
import logging
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
from sqlalchemy import select

from surveilx.config import settings
from surveilx.controller import Candidate, Scheduler
from surveilx.database import Camera, Incident, Record, audit, transaction
from surveilx.evidence import delete_bundle, save_bundle
from surveilx.hardware import PowerGovernor, probe
from surveilx.incidents import create_incident
from surveilx.security import cipher
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
        self.tracker = Tracker()
        self.buffer = deque(maxlen=30)
        self.candidate_since = None
        self.last_incident = 0.0
        self.previous_gray = None
        self.entity_history = deque(maxlen=8)
        self.secondary_result = None
        self.motion = 0.0
        self.brightness = 0.0
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"capture-{self.id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
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
                frame = cv2.resize(frame, (640, max(128, int(height * 640 / width))))
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
                "calibration": "uncalibrated",
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
        self.previous_sva = None
        self.sva_lock = threading.Lock()
        if settings.detector_path:
            try:
                self.yolo = YOLODetector(settings.detector_path)
            except Exception as exc:
                self.error = f"Detector load failed: {type(exc).__name__}; HOG fallback active"
        self.thread = threading.Thread(target=self.run, daemon=True, name="global-inference")

    def activate_sva(self, directory):
        from surveilx.expert import SVAExpert

        expert = SVAExpert(directory)
        with self.sva_lock:
            self.previous_sva, self.sva = self.sva, expert

    def rollback_sva(self):
        with self.sva_lock:
            self.sva, self.previous_sva = self.previous_sva, None

    def start(self):
        self.sync_cameras()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        for capture in list(self.cameras.values()):
            capture.stop()

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
            payload = json.loads(cipher().decrypt(path.read_bytes()))
            with transaction() as session:
                # Evidence key is stable across retries; avoid duplicate incident creation.
                existing = session.scalar(
                    select(Incident).where(Incident.evidence_key == payload["evidence_key"])
                )
                if not existing:
                    create_incident(session, **payload)
            path.unlink()

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
            if capture.frame is None:
                return
            frame = capture.frame.copy()
            capture.consumed = capture.sequence
        started = time.perf_counter()
        detector = self.synthetic_detector if capture.synthetic else (self.yolo or self.baseline)
        try:
            detections = detector.infer(frame, self.power["resolution"])
        except Exception as exc:
            self.error = f"Expert failed: {type(exc).__name__}; baseline fallback"
            detector = self.baseline
            detections = detector.infer(frame, 320)
        detections = capture.tracker.update(detections)
        outputs = [d.json() for d in detections]
        capture.entity_history.append((frame, outputs))
        # Secondary computation only on active candidates and compatible, explicitly activated experts.
        with self.sva_lock:
            expert = self.sva
        if expert and capture.candidate_since is not None:
            capture.secondary_result = expert.infer(
                list(capture.entity_history), capture.synthetic, capture.environment
            )
        elapsed = (time.perf_counter() - started) * 1000
        self.latencies.append(elapsed)
        now = time.monotonic()
        with capture.lock:
            capture.outputs = outputs
            capture.detector = detector.name
            capture.latency_ms = elapsed
            capture.last_service = now
        occupied = False
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            for zone in capture.zones:
                left, top, right, bottom = zone
                occupied |= left <= center[0] <= right and top <= center[1] <= bottom
        if not occupied:
            capture.candidate_since = None
        elif capture.candidate_since is None:
            capture.candidate_since = now
        elif (
            now - capture.candidate_since >= settings.confirmation_seconds
            and now - capture.last_incident >= settings.cooldown_seconds
        ):
            details = {
                "detections": capture.outputs,
                "model": detector.name,
                "calibration": "uncalibrated",
                "decision": self.decision,
                "power": self.power,
                "synthetic": capture.synthetic,
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
        with transaction() as session:
            session.add(
                Record(
                    kind="decision",
                    payload={
                        "timestamp": time.time(),
                        "camera_id": capture.id,
                        "action": detector.name,
                        "resolution": self.power["resolution"],
                        "latency_ms": elapsed,
                        "state": {"motion": capture.motion, "brightness": capture.brightness},
                        "result": capture.outputs,
                        "secondary": capture.secondary_result,
                        "calibration": "uncalibrated",
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
                    candidates, self.power["budget_ms"], settings.coverage_seconds
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
                    self.sync_cameras()
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
