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

from surveilx.detection_preferences import DetectionSelection, infer_selected, model_enabled
from surveilx.config import settings
from surveilx.controller import Candidate, Scheduler
from surveilx.database import Camera, Incident, ModelVersion, Record, audit, transaction
from surveilx.drift import drift_monitor
from surveilx.evidence import delete_bundle, save_bundle
from surveilx.hardware import PowerGovernor, probe
from surveilx.preview import PreviewEncoder
from surveilx.incidents import create_incident
from surveilx.notifications import process_due
from surveilx.security import cipher
from surveilx.policies import PolicyEngine
from surveilx.vision import HOGDetector, SyntheticDetector, Tracker, YOLODetector, generated_frame, redact

logger = logging.getLogger("surveilx")


class IntervalGate:
    """Wall-clock maintenance cadence independent of inference tick frequency."""

    def __init__(self):
        self.next_due = {}

    def due(self, name, seconds, now):
        if now < self.next_due.get(name, float("-inf")):
            return False
        self.next_due[name] = now + seconds
        return True


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
        self.preview = PreviewEncoder()
        self.consumed = 0
        self.inference_count = 0
        self.selection_work = {}
        self.dropped = 0
        self.reconnects = 0
        self.status = "connecting"
        self.fps = 5
        self.last_service = time.monotonic()
        self.latency_ms = 50.0
        self.outputs = []
        self.detection_selection = DetectionSelection()
        self.preference_lock = threading.RLock()
        self.detector = "pending"
        self.calibration = "uncalibrated"
        self.category_routes = {}
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
                    # Keep analysis/evidence sampling independent of preview FPS.
                    if not self.buffer or self.timestamp - self.buffer[-1][0] >= 1 / min(8, self.fps):
                        self.brightness = float(gray.mean() / 255)
                        self.motion = (
                            float(np.mean(cv2.absdiff(gray, self.previous_gray) > 20))
                            if self.previous_gray is not None else 0
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
                "inference_count": self.inference_count,
                "selection_work": dict(self.selection_work),
                "frame_age_seconds": round(max(0, time.time() - self.timestamp), 3),
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
                "category_routes": self.category_routes,
                "secondary_evidence": self.secondary_result,
                "risk": "review" if self.candidate_since else "no active zone observation",
            }


class Runtime:
    def __init__(self):
        self.cameras = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.governor = PowerGovernor(settings.power_profile)
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
        self.category_routers = {}
        self.event_failed_until = {}
        self.started = False
        if settings.detector_path:
            try:
                self.yolo = YOLODetector(settings.detector_path)
            except Exception as exc:
                self.error = f"Detector load failed: {type(exc).__name__}; HOG fallback active"
        self.drift_thread = None
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
            self.category_routers.clear()
            if task == "detection":
                self.learned_detector = expert
            elif task == "event":
                self.sva = expert
            if expert is None:
                self.experts.pop(task, None)
            else:
                self.experts[task] = expert
            self.active_models[task] = model_id
            live_versions = {getattr(value, "version", "default") for value in self.experts.values()}
            self.event_failed_until = {k: v for k, v in self.event_failed_until.items() if k in live_versions}

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
                        selection = session.get(Record, f"detection-selection:{camera.id}")
                        if selection:
                            self.cameras[camera.id].detection_selection = DetectionSelection.model_validate(selection.payload)
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
            "live_processing": self.started,
            "hardware": self.hardware,
            "power": self.power,
            "decision": self.decision,
            "heartbeat": self.heartbeat,
            "error": self.error,
            "active_models": dict(self.active_models),
            "category_routing": {str(key): router.status() for key, router in self.category_routers.items()},
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

    def set_detection_selection(self, camera_id, selection):
        capture = self.cameras.get(camera_id)
        if capture:
            with capture.preference_lock, capture.lock:
                capture.detection_selection = selection
                capture.outputs = []
                capture.secondary_result = None
                capture.category_routes = {}
                capture.selection_work = {}
                capture.tracker = Tracker()
                capture.entity_history.clear()
                capture.scene_history.clear()
                capture.scene_times.clear()
                capture.expert_states.clear()
                capture.candidate_since = None
                capture.last_detection_signature = ()
                capture.policy_engine = PolicyEngine(capture.policy_engine.policy.model_dump())
                capture.last_service = time.monotonic()

    def infer(self, capture):
        with capture.preference_lock:
            if not capture.stop_event.is_set():
                return self._infer(capture)

    def has_detection_work(self, capture):
        """Keep intentionally idle cameras out of the global inference budget."""
        selection = capture.detection_selection
        if not selection.disabled_labels:
            return True
        baseline = self.synthetic_detector if capture.synthetic else (self.yolo or self.baseline)
        if model_enabled(baseline, selection):
            return True
        with self.sva_lock:
            values = [self.learned_detector, self.sva, *self.experts.values()]
            for value in values:
                if value is None or not model_enabled(value, selection):
                    continue
                manifest = getattr(value, "manifest", {})
                if not manifest or (capture.synthetic == bool(manifest.get("synthetic", False))
                        and ((capture.synthetic and value is self.learned_detector)
                             or capture.environment == manifest.get("domain"))):
                    return True
        return False

    def _infer(self, capture):
        with capture.lock:
            if capture.frame is None or capture.sequence <= capture.consumed:
                return
            frame = capture.frame.copy()
            captured_at = capture.timestamp
            capture.consumed = capture.sequence
        started = time.perf_counter()
        with self.sva_lock:
            learned = self.learned_detector
            if learned and not model_enabled(learned, capture.detection_selection):
                learned = None
            specialists = [
                (key, value)
                for key, value in sorted(self.experts.items())
                if key.startswith("detection:")
                and model_enabled(value, capture.detection_selection)
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
        # Only models already loaded through explicit deployment enter automatic routing.
        # Missing/incompatible validation evidence preserves the historical slot behavior.
        deployed = ([("detection", learned)] if learned and selected_key == "detection" else []) + specialists
        if specialists and learned and all(value is not learned for _, value in deployed):
            if (capture.synthetic == bool(learned.manifest.get("synthetic", False))
                    and capture.environment == learned.manifest.get("domain")):
                deployed.append(("detection", learned))
        if deployed:
            from surveilx.model_selection import evaluation_for
            from surveilx.model_router import CategoryRouter
            signature = tuple(sorted((slot, value.version) for slot, value in deployed))
            router = self.category_routers.get(signature)
            if router is None:
                candidates = [{"version": value.version, "manifest": value.manifest, "expert": value,
                    "runtime_device": getattr(value, "device", None),
                    "evaluation": evaluation_for(settings.data_dir, value.version, value.manifest)}
                    for _, value in deployed]
                router = CategoryRouter(candidates if all(c["evaluation"] for c in candidates) else [])
                if router.routes:
                    self.category_routers[signature] = router
                    while len(self.category_routers) > 32:
                        self.category_routers.pop(next(iter(self.category_routers)))
            if router.routes:
                detector, selected_key = router, "category-routing"
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
        # Secondary computation only on active candidates and compatible, explicitly activated experts.
        with self.sva_lock:
            expert = self.sva
            if expert and not model_enabled(expert, capture.detection_selection):
                expert = None
            manifest = getattr(expert, "manifest", {})
            if manifest and (capture.environment != manifest.get("domain")
                    or capture.synthetic != bool(manifest.get("synthetic", False))):
                expert = None
            event_slot = "event:default"
            events = [
                (name, value)
                for name, value in sorted(self.experts.items())
                if name.startswith("event:")
                and model_enabled(value, capture.detection_selection)
                and capture.environment == value.manifest.get("domain")
                and capture.synthetic == bool(value.manifest.get("synthetic", False))
            ]
            if events:
                event_slot, expert = events[capture.expert_cursor % len(events)]
                from surveilx.model_selection import evaluation_for, rank_categories, contract
                candidates = [{"version": value.version, "manifest": value.manifest,
                    "evaluation": evaluation_for(settings.data_dir, value.version, value.manifest)}
                    for _, value in events]
                # Separate scene/entity contracts and different event corpora are never ranked together.
                if len({contract(c["manifest"]) for c in candidates}) == 1 and all(c["evaluation"] for c in candidates):
                    ranked = rank_categories(candidates)
                    scores = {}
                    for route in ranked:
                        if route["category"] in {"normal", "background"} or not capture.detection_selection.enabled(route["category"]):
                            continue
                        for candidate in [route["selected"], *route["fallbacks"]]:
                            scores.setdefault(candidate["version"], []).append(candidate["score"])
                    if scores:
                        winner = max(scores, key=lambda version: (float(np.mean(scores[version])), version))
                        event_slot, expert = next((slot, value) for slot, value in events if value.version == winner)
        if not specialists:
            capture.expert_cursor += 1
        capture.secondary_result = None
        scene_expert = bool(expert and getattr(expert, "scene", False))
        skipped_primary = False
        try:
            skipped_primary = not model_enabled(detector, capture.detection_selection)
            detections = infer_selected(detector, frame, self.power["resolution"], capture.detection_selection)
        except Exception as exc:
            self.error = f"Expert failed: {type(exc).__name__}; baseline fallback"
            detector = self.synthetic_detector if capture.synthetic else self.baseline
            selected_key = "fallback"
            skipped_primary = not model_enabled(detector, capture.detection_selection)
            detections = infer_selected(detector, frame, 320, capture.detection_selection)
            capture.tracker = Tracker()
            capture.entity_history.clear()
            capture.candidate_since = None
            capture.expert_key = ("fallback", detector.name)
        detections = capture.tracker.update(detections) if not skipped_primary else []
        outputs = [d.json() for d in detections]
        temporal_candidates = [value for _, value in events] + ([expert] if expert else [])
        needs_scene = any(getattr(value, "scene", False) for value in temporal_candidates)
        needs_entity = any(not getattr(value, "scene", False) for value in temporal_candidates)
        if needs_scene or needs_entity:
            compressed, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            history_frame = encoded.tobytes() if compressed else frame
            if needs_entity:
                capture.entity_history.append((history_frame, outputs))
            if needs_scene:
                capture.scene_history.append((history_frame, []))
                capture.scene_times.append(captured_at)
        if not needs_entity:
            capture.entity_history.clear()
        if not needs_scene:
            capture.scene_history.clear()
            capture.scene_times.clear()
        capture.selection_work = {
            "primary_skipped": skipped_primary,
            "enabled_detection_specialists": len(specialists),
            "enabled_event_models": len({id(value) for value in temporal_candidates}),
            "history_encoded": needs_scene or needs_entity,
        }
        if skipped_primary and not temporal_candidates:
            # Preview capture/encoding is independent. With no requested work,
            # consume the frame without tracking, policy, evidence or DB writes.
            with capture.lock:
                capture.outputs = []
                capture.secondary_result = None
                capture.candidate_since = None
                capture.category_routes = {}
                capture.detector = "detection-disabled"
                capture.calibration = "not_applicable"
                capture.latency_ms = (time.perf_counter() - started) * 1000
                capture.last_service = time.monotonic()
            return
        if expert and (capture.candidate_since is not None or scene_expert):
            try:
                if self.event_failed_until.get(getattr(expert, "version", "default"), 0) > time.monotonic():
                    raise RuntimeError("Temporal expert is in recovery cooldown")
                capture.secondary_result = expert.infer(
                    list(capture.scene_history if scene_expert else capture.entity_history),
                    capture.synthetic,
                    capture.environment,
                    **({"timestamps": list(capture.scene_times)} if scene_expert else {}),
                )
            except Exception as exc:
                capture.secondary_result = {"decision": "abstain", "reason": "Secondary expert failed"}
                self.error = f"Secondary expert failed: {type(exc).__name__}; primary detections retained"
                from surveilx.model_selection import contract
                failed_version = getattr(expert, "version", "default")
                if self.event_failed_until.get(failed_version, 0) <= time.monotonic():
                    self.event_failed_until[failed_version] = time.monotonic() + 15
                for fallback_slot, fallback in events:
                    if (fallback is expert or contract(fallback.manifest) != contract(expert.manifest)
                            or self.event_failed_until.get(fallback.version, 0) > time.monotonic()):
                        continue
                    try:
                        fallback_scene = bool(getattr(fallback, "scene", False))
                        capture.secondary_result = fallback.infer(
                            list(capture.scene_history if fallback_scene else capture.entity_history),
                            capture.synthetic, capture.environment,
                            **({"timestamps": list(capture.scene_times)} if fallback_scene else {}))
                        event_slot, expert = fallback_slot, fallback
                        self.event_failed_until.pop(fallback.version, None)
                        break
                    except Exception:
                        self.event_failed_until[fallback.version] = time.monotonic() + 15
        capture.secondary_result = capture.detection_selection.event(capture.secondary_result)
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
            capture.category_routes = detector.status() if selected_key == "category-routing" else {}
            if capture.category_routes:
                capture.calibration = "component_calibrated; pooled_reliability_unvalidated"
            capture.latency_ms = elapsed
            capture.last_service = now
            capture.inference_count += int(not skipped_primary)
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
                "category_routes": capture.category_routes,
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
        for observation in capture.policy_engine.evaluate(outputs, now, capture.expert_key, capture.detection_selection.disabled_labels):
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
        for observation in capture.policy_engine.evaluate_event(capture.secondary_result, now, event_key, capture.detection_selection.disabled_labels):
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

    def request_drift_check(self):
        # Statistical permutation tests must never pause live inference. Keep one
        # worker and skip overlapping requests rather than building a work queue.
        if self.drift_thread is not None and self.drift_thread.is_alive():
            return False

        def evaluate():
            try:
                drift_monitor.evaluate()
            except Exception as exc:
                self.error = f"Drift check failed: {type(exc).__name__}"
                logger.exception("Background drift check failed")

        self.drift_thread = threading.Thread(target=evaluate, daemon=True, name="drift-monitor")
        self.drift_thread.start()
        return True

    def run(self):
        intervals = IntervalGate()
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                if intervals.due("hardware", 1.0, start):
                    hardware = probe()
                    self.hardware = hardware.json()
                    recent_latency = float(np.percentile(list(self.latencies)[-20:], 95)) if self.latencies else 0
                    self.power = self.governor.update(hardware, recent_latency)
                with self.lock:
                    captures = list(self.cameras.values())
                candidates = []
                for capture in captures:
                    capture.fps = self.power["capture_fps"]
                    has_work = self.has_detection_work(capture)
                    if not has_work:
                        with capture.lock:
                            capture.consumed = capture.sequence
                            capture.last_service = time.monotonic()
                            capture.latency_ms = 0.0
                            capture.detector = "detection-disabled"
                            capture.selection_work = {"primary_skipped": True,
                                "enabled_detection_specialists": 0, "enabled_event_models": 0,
                                "history_encoded": False}
                    age = time.monotonic() - capture.last_service
                    # Floor is cold-start reservation, not a fabricated benchmark.
                    candidates.append(
                        Candidate(
                            capture.id,
                            capture.priority,
                            age,
                            max(10, capture.latency_ms * 1.25),
                            has_work and capture.status == "online" and capture.sequence > capture.consumed,
                        )
                    )
                self.decision = self.scheduler.allocate(
                    candidates, min(900.0, self.power["budget_ms"] * settings.runtime_budget_scale)
                    * min(1.0, settings.epoch_seconds),
                    settings.coverage_seconds, settings.epoch_seconds
                )
                for camera_id in self.decision["selected"]:
                    capture = next((c for c in captures if c.id == camera_id), None)
                    if capture:
                        try:
                            self.infer(capture)
                        except Exception as exc:
                            self.error = f"Inference/persistence failure: {type(exc).__name__}"
                if intervals.due("maintenance", 10.0, start):
                    self.flush_spool()
                    with transaction() as session:
                        session.add(Record(kind="hardware", payload=self.hardware | {"power": self.power}))
                        process_due(session)
                    self.sync_cameras()
                if intervals.due("drift", 60.0, start):
                    self.request_drift_check()
                if intervals.due("retention", 600.0, start):
                    self.retention()
                self.heartbeat = time.time()
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
