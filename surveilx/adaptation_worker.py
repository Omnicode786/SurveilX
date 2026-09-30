"""Single-process, opt-in retraining from independently approved annotations."""

import threading
import time
import uuid

from pydantic import BaseModel, Field
from sqlalchemy import select

from surveilx.adaptation import DatasetBuild, build_dataset, priority
from surveilx.database import Record, audit, transaction
from surveilx.jobs import jobs


class AdaptationPolicy(BaseModel):
    enabled: bool = False
    base_dataset: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    architecture: str = Field(default="auto", pattern="^(auto|scratch|yolo_rai)$")
    epochs: int = Field(default=5, ge=1, le=500)
    minimum_approved: int = Field(default=10, ge=1, le=500)
    max_new: int = Field(default=100, ge=1, le=500)
    replay_limit: int = Field(default=500, ge=10, le=10000)


class AdaptationWorker:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None

    def status(self):
        with transaction() as session:
            row = session.get(Record, "adaptation-policy")
            return dict(row.payload) if row else {"enabled": False, "state": "not_configured"}

    def configure(self, policy, actor):
        from surveilx.config import settings
        from training.datasets import validate_manifest

        manifest, _ = validate_manifest(
            settings.data_dir / "datasets" / policy.base_dataset / "manifest.json"
        )
        if policy.minimum_approved > policy.max_new:
            raise ValueError("minimum_approved cannot exceed max_new")
        if policy.architecture == "yolo_rai" and manifest.get("task") != "detection":
            raise ValueError("YOLO policy requires a detection dataset")
        with self.lock, transaction() as session:
            row = session.get(Record, "adaptation-policy")
            old = row.payload if row else {}
            if old.get("pending") and old.get("base_dataset") != policy.base_dataset:
                raise ValueError("Finish the pending generation before changing its base dataset")
            payload = {
                **old,
                **policy.model_dump(),
                "task": manifest.get("task", "event"),
                "domain": manifest["domain"],
                "synthetic": manifest.get("synthetic", False),
                "configured_by": actor,
                "last_error": None,
                "blocked_annotation_ids": [],
                "state": "waiting_for_reviewed_data" if policy.enabled else "disabled",
            }
            payload["consumed_annotation_ids"] = sorted(
                set(old.get("consumed_annotation_ids", []))
                | set(manifest.get("adaptation", {}).get("consumed_annotation_ids", []))
            )
            if row:
                row.payload = payload
            else:
                session.add(Record(id="adaptation-policy", kind="adaptation_policy", payload=payload))
            audit(session, actor, "adaptation_policy_configured", "adaptation-policy", enabled=policy.enabled)
        return payload

    def update(self, **changes):
        with transaction() as session:
            row = session.get(Record, "adaptation-policy")
            if row:
                row.payload = {**row.payload, **changes}

    def tick(self):
        with self.lock:
            policy = self.status()
            pending = policy.get("pending")
            if pending:
                with transaction() as session:
                    job = session.get(Record, pending["job_id"])
                    state = job.payload.get("state") if job else "missing"
                if state in {"queued", "running"}:
                    return
                if state == "completed":
                    consumed = sorted(
                        set(policy.get("consumed_annotation_ids", [])) | set(pending["annotation_ids"])
                    )
                    self.update(
                        pending=None,
                        base_dataset=pending["dataset"],
                        consumed_annotation_ids=consumed,
                        last_completed_job=pending["job_id"],
                        state="waiting_for_reviewed_data",
                    )
                else:
                    self.update(
                        pending=None,
                        state="attention_required",
                        blocked_annotation_ids=pending["annotation_ids"],
                        last_error=f"Generation {pending['job_id']} ended as {state}; review logs and save policy to retry",
                    )
                return
            if not policy.get("enabled") or jobs.running:
                return
            consumed = set(policy.get("consumed_annotation_ids", []))
            with transaction() as session:
                records = list(session.scalars(select(Record).where(Record.kind == "annotation")))
            compatible = [
                r
                for r in records
                if r.id not in consumed
                and r.payload.get("state") == "approved"
                and r.payload.get("task") == policy["task"]
                and r.payload.get("domain") == policy["domain"]
                and r.payload.get("synthetic") == policy["synthetic"]
            ]
            if len(compatible) < policy["minimum_approved"]:
                self.update(state="waiting_for_reviewed_data", available_approved=len(compatible))
                return
            compatible.sort(key=lambda r: (-priority(r.payload), r.timestamp, r.id))
            ids = [r.id for r in compatible]
            if policy.get("blocked_annotation_ids"):
                return
            name = f"adapt-{uuid.uuid4().hex[:16]}"
            try:
                built = build_dataset(
                    DatasetBuild(
                        name=name,
                        base_dataset=policy["base_dataset"],
                        annotation_ids=ids[:500],
                        max_new=policy["max_new"],
                        replay_limit=policy["replay_limit"],
                    ),
                    policy["configured_by"],
                )
                parent = policy.get("last_completed_job") if policy["architecture"] != "yolo_rai" else None
                job_id = jobs.start(
                    name,
                    policy["epochs"],
                    policy["configured_by"],
                    policy["architecture"],
                    parent,
                )
                self.update(
                    pending={
                        "job_id": job_id,
                        "dataset": name,
                        "annotation_ids": built["annotation_ids"],
                        "initialize_from": parent,
                    },
                    state="training_candidate",
                    last_started=time.time(),
                    last_error=None,
                )
            except Exception as exc:
                self.update(state="attention_required", last_error=str(exc), blocked_annotation_ids=ids)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()

        def loop():
            while not self.stop_event.is_set():
                try:
                    self.tick()
                except Exception:
                    # Keep the monitor alive across transient DB failures; no deployment is performed here.
                    pass
                self.stop_event.wait(15)

        self.thread = threading.Thread(target=loop, daemon=True, name="reviewed-adaptation")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)


adaptation_worker = AdaptationWorker()
