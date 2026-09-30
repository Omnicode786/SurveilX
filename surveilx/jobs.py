import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import ModelVersion, Record, audit, transaction
from training.datasets import digest


def retention_report(result, current_dataset, parent_directory):
    """Compare stored metrics only after proving held-out samples were copied unchanged."""
    parent = json.loads((parent_directory / "manifest.json").read_text(encoding="utf-8"))
    base_name = current_dataset.get("adaptation", {}).get("base_dataset")
    if not base_name:
        raise ValueError("Continued training requires adaptation base-dataset lineage")
    base_path = settings.data_dir / "datasets" / base_name / "manifest.json"
    if digest(base_path) != parent.get("dataset_sha256"):
        raise ValueError("Continuation parent was not trained on the declared base dataset")
    base = json.loads(base_path.read_text(encoding="utf-8"))

    def heldout(document):
        return sorted(
            json.dumps(sample, sort_keys=True, separators=(",", ":"))
            for sample in document["samples"]
            if sample["split"] != "train"
        )

    if heldout(base) != heldout(current_dataset):
        raise ValueError("Validation/calibration/test samples changed during continued training")
    score = "map50" if result.get("task") == "detection" else "accuracy"
    before, after = parent.get("metrics", {}).get(score), result.get("metrics", {}).get(score)
    if not all(isinstance(value, (int, float)) for value in (before, after)):
        raise ValueError("Comparable parent and candidate retention metrics are required")
    return {
        "parent_version": parent_directory.name,
        "parent_weights_sha256": parent["weights_sha256"],
        "metric": score,
        "parent_score": before,
        "candidate_score": after,
        "delta": after - before,
        "heldout_samples_unchanged": True,
        "measurement": "stored parent and current metrics on byte-identical copied held-out samples",
        "deployment_effect": "reported only; independent acceptance remains required",
    }


class TrainingJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False

    def recover_interrupted(self):
        """Single-server recovery: an in-process worker cannot survive a server restart."""
        with self.lock:
            if self.running:
                return
            with transaction() as session:
                for record in session.scalars(select(Record).where(Record.kind == "experiment")):
                    if record.payload.get("state") in {"queued", "running"}:
                        record.payload = {
                            **record.payload,
                            "state": "interrupted",
                            "finished": time.time(),
                            "error": "Server restarted before the worker recorded completion; review process.log before retry",
                        }

    def start(self, dataset, epochs, actor, architecture="auto", initialize_from=None):
        with self.lock:
            if self.running:
                raise ValueError("A training job is already running")
            self.running = True
        job_id = str(uuid.uuid4())
        try:
            with transaction() as session:
                session.add(
                    Record(
                        id=job_id,
                        kind="experiment",
                        payload={
                            "state": "queued",
                            "dataset": dataset,
                            "epochs": epochs,
                            "architecture": architecture,
                            "initialize_from": initialize_from,
                            "queued": time.time(),
                        },
                    )
                )
                audit(session, actor, "training_queued", job_id, dataset=dataset, architecture=architecture)
            threading.Thread(
                target=self.run,
                args=(job_id, dataset, epochs, actor, architecture, initialize_from),
                daemon=True,
            ).start()
        except Exception:
            with self.lock:
                self.running = False
            with transaction() as session:
                record = session.get(Record, job_id)
                if record:
                    record.payload = {
                        **record.payload,
                        "state": "failed",
                        "error": "Could not start worker",
                        "finished": time.time(),
                    }
            raise
        return job_id

    def run(self, job_id, dataset, epochs, actor, architecture="auto", initialize_from=None):
        directory = (settings.data_dir / "runs" / job_id).resolve()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with transaction() as session:
                record = session.get(Record, job_id)
                record.payload = {**record.payload, "state": "running", "started": time.time()}
                audit(session, actor, "training_started", job_id, dataset=dataset)
            manifest = (settings.data_dir / "datasets" / dataset / "manifest.json").resolve()
            dataset_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            task = dataset_manifest.get("task", "event")
            if architecture == "yolo_rai":
                if task != "detection":
                    raise ValueError("YOLO training requires a detection dataset")
                module = "training.yolo_pipeline"
            else:
                module = "training.detection_pipeline" if task == "detection" else "training.pipeline"
            command = [
                sys.executable,
                "-m",
                module,
                "train",
                str(manifest),
                str(directory),
                "--epochs",
                str(epochs),
            ]
            if initialize_from:
                if architecture == "yolo_rai":
                    raise ValueError("Automatic YOLO continuation is not yet validated")
                parent = (settings.data_dir / "runs" / initialize_from).resolve()
                if not parent.is_relative_to((settings.data_dir / "runs").resolve()):
                    raise ValueError("Continuation parent escaped the run registry")
                command.extend(["--initialize-from", str(parent)])
            with (directory / "process.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=log,
                    timeout=6 * 3600,
                    cwd=Path(__file__).resolve().parents[1],
                )
            if completed.returncode:
                raise RuntimeError("Training failed; inspect the run process.log")
            result = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(result, dict) or not isinstance(result.get("name"), str):
                raise ValueError("Training process did not produce a valid model manifest")
            if initialize_from:
                result["retention"] = retention_report(result, dataset_manifest, parent)
                (directory / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            with transaction() as session:
                record = session.get(Record, job_id)
                record.payload = {
                    **record.payload,
                    "state": "completed",
                    "finished": time.time(),
                    "result": result,
                }
                session.add(ModelVersion(name=result["name"], version=job_id, manifest=result))
                audit(session, actor, "training_calibration_completed", job_id)
        except Exception as exc:
            with transaction() as session:
                record = session.get(Record, job_id)
                if record:
                    record.payload = {
                        **record.payload,
                        "state": "failed",
                        "finished": time.time(),
                        "error": str(exc),
                    }
                    audit(session, actor, "training_failed", job_id, error=type(exc).__name__)
        finally:
            with self.lock:
                self.running = False


jobs = TrainingJobs()
