"""Run frozen-model evaluations and bind acceptance to exact artifacts and policies."""

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import ModelVersion, Record, audit, transaction
from training.datasets import digest


class AcceptancePolicy(BaseModel):
    min_samples: int = Field(default=100, ge=10, le=1000000)
    min_groups: int = Field(default=5, ge=2, le=100000)
    min_per_class: int = Field(default=20, ge=5, le=100000)
    min_ap50: float = Field(default=0.8, ge=0, le=1)
    min_recall_lower: float = Field(default=0.8, ge=0, le=1)
    min_precision_lower: float = Field(default=0.8, ge=0, le=1)
    min_accuracy_lower: float = Field(default=0.8, ge=0, le=1)
    max_ece: float = Field(default=0.08, ge=0, le=1)
    max_wall_ms: float = Field(default=500, gt=0, le=60000)
    max_regression: float = Field(default=0.02, ge=0, le=0.2)


class AcceptanceInput(BaseModel):
    dataset: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    baseline_model_id: str | None = None
    policy: AcceptancePolicy = Field(default_factory=AcceptancePolicy)


def references_for(models):
    paths = list((settings.data_dir / "datasets").glob("*/manifest.json"))
    by_hash = {digest(p): p for p in paths}
    if any(not model.manifest.get("dataset_sha256") for model in models):
        raise ValueError("Local calibrated training lineage is required before production acceptance")
    pending = [model.manifest["dataset_sha256"] for model in models]
    resolved = {}
    while pending:
        checksum = pending.pop()
        if checksum in resolved:
            continue
        if checksum not in by_hash:
            raise ValueError("An original or ancestor training manifest is missing or modified")
        path = by_hash[checksum]
        resolved[checksum] = str(path.resolve())
        parent = json.loads(path.read_text()).get("adaptation", {}).get("base_sha256")
        if parent:
            pending.append(parent)
    return list(resolved.values())


class EvaluationJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False

    def recover(self):
        with transaction() as session:
            for row in session.scalars(select(Record).where(Record.kind == "acceptance")):
                if row.payload.get("state") in {"queued", "running"}:
                    row.payload = {
                        **row.payload,
                        "state": "interrupted",
                        "error": "Server restarted during evaluation",
                    }

    def start(self, model_id, body, actor):
        with self.lock:
            if self.running:
                raise ValueError("An acceptance evaluation is already running")
            self.running = True
        try:
            with transaction() as session:
                model = session.get(ModelVersion, model_id)
                baseline = (
                    session.get(ModelVersion, body.baseline_model_id) if body.baseline_model_id else None
                )
                if not model or (body.baseline_model_id and not baseline):
                    raise ValueError("Unknown candidate or baseline")
                models = [model, baseline] if baseline else [model]
                if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", m.version) for m in models):
                    raise ValueError("Invalid model artifact version")
                request = {
                    "run": str((settings.data_dir / "runs" / model.version).resolve()),
                    "baseline": str((settings.data_dir / "runs" / baseline.version).resolve())
                    if baseline
                    else None,
                    "dataset": str(
                        (settings.data_dir / "datasets" / body.dataset / "manifest.json").resolve()
                    ),
                    "references": references_for(models),
                    "policy": body.policy.model_dump(),
                }
                job_id = str(uuid.uuid4())
                directory = (settings.data_dir / "evaluations" / job_id).resolve()
                directory.mkdir(parents=True)
                request_path = directory / "request.json"
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                session.add(
                    Record(
                        id=job_id,
                        kind="acceptance",
                        payload={
                            "state": "queued",
                            "model_id": model_id,
                            "baseline_model_id": body.baseline_model_id,
                            "dataset": body.dataset,
                            "policy": body.policy.model_dump(),
                            "requested_by": actor,
                            "request_sha256": digest(request_path),
                            "weights_sha256": model.manifest["weights_sha256"],
                        },
                    )
                )
                audit(
                    session,
                    actor,
                    "acceptance_requested",
                    job_id,
                    model_id=model_id,
                    policy=body.policy.model_dump(),
                )
            threading.Thread(target=self.run, args=(job_id, directory, actor), daemon=True).start()
            return job_id
        except Exception:
            self.running = False
            raise

    def run(self, job_id, directory, actor):
        try:
            with transaction() as session:
                row = session.get(Record, job_id)
                row.payload = {**row.payload, "state": "running", "started": time.time()}
            with (directory / "process.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "training.acceptance",
                        str(directory / "request.json"),
                        str(directory),
                    ],
                    stdout=log,
                    stderr=log,
                    timeout=6 * 3600,
                    cwd=Path(__file__).resolve().parents[1],
                )
            if result.returncode:
                raise ValueError(
                    "Evaluation failed; inspect its process.log for leakage, provenance or inference errors"
                )
            report = json.loads((directory / "report.json").read_text())
            with transaction() as session:
                row = session.get(Record, job_id)
                model = session.get(ModelVersion, row.payload["model_id"])
                if (
                    report["request_sha256"] != row.payload["request_sha256"]
                    or report["weights_sha256"] != model.manifest["weights_sha256"]
                ):
                    raise ValueError("Evaluation artifact binding changed")
                row.payload = {**row.payload, "state": "completed", "report": report, "finished": time.time()}
                # Record evidence; an administrator must explicitly accept it before production eligibility.
                audit(session, actor, "acceptance_evaluated", job_id, passed=report["passed"])
        except Exception as exc:
            with transaction() as session:
                row = session.get(Record, job_id)
                row.payload = {**row.payload, "state": "failed", "error": str(exc), "finished": time.time()}
        finally:
            with self.lock:
                self.running = False


def accept_report(report_id, actor):
    with transaction() as session:
        row = session.get(Record, report_id)
        if not row or row.kind != "acceptance" or row.payload.get("state") != "completed":
            raise ValueError("Completed acceptance report required")
        report = row.payload["report"]
        if not report.get("deployment_eligible"):
            raise ValueError("Acceptance criteria failed or data is synthetic")
        model = session.get(ModelVersion, row.payload["model_id"])
        if not model or not report.get("passed") or report.get("synthetic"):
            raise ValueError("A passing real-domain report and existing model are required")
        actual = digest(settings.data_dir / "runs" / model.version / "weights.pt")
        if actual != report["weights_sha256"] or model.manifest["weights_sha256"] != actual:
            raise ValueError("Model changed since evaluation")
        manifest_hash = digest(settings.data_dir / "runs" / model.version / "manifest.json")
        if manifest_hash != report.get("artifact_manifest_sha256") or report["domain"] != model.manifest.get(
            "domain"
        ):
            raise ValueError("Model configuration or calibration changed since evaluation")
        model.manifest = {
            **model.manifest,
            "deployment_eligible": True,
            "acceptance": {
                "report_id": report_id,
                "weights_sha256": actual,
                "artifact_manifest_sha256": manifest_hash,
                "domain": report["domain"],
                "dataset_sha256": report["dataset_sha256"],
                "approved_by": actor,
                "approved_at": time.time(),
            },
        }
        row.payload = {**row.payload, "approved_by": actor, "approved_at": time.time()}
        audit(session, actor, "acceptance_approved", report_id, model_id=model.id)
        return {"model_id": model.id, "deployment_eligible": True, "stage": model.stage}


def verify_approved_artifact(model, session):
    """Production and restart checks must bind approval to weights AND calibration/configuration."""
    acceptance = model.manifest.get("acceptance", {})
    record = session.get(Record, acceptance.get("report_id", ""))
    if (
        not record
        or record.kind != "acceptance"
        or not record.payload.get("approved_by")
        or record.payload.get("model_id") != model.id
    ):
        raise ValueError("An approved acceptance report for this model is required")
    report = record.payload.get("report", {})
    if not report.get("passed") or not report.get("deployment_eligible") or report.get("synthetic"):
        raise ValueError("Acceptance report is not eligible for production")
    root = settings.data_dir / "runs" / model.version
    if (
        digest(root / "weights.pt") != report.get("weights_sha256")
        or digest(root / "manifest.json") != report.get("artifact_manifest_sha256")
        or report.get("domain") != model.manifest.get("domain")
    ):
        raise ValueError("Production artifact differs from the accepted model or calibration")


evaluation_jobs = EvaluationJobs()
