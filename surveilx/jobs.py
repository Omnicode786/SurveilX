import json
import subprocess
import sys
import threading
import time
import uuid

from surveilx.config import settings
from surveilx.database import ModelVersion, Record, audit, transaction


class TrainingJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False

    def start(self, dataset, epochs, actor):
        with self.lock:
            if self.running:
                raise ValueError("A training job is already running")
            self.running = True
        job_id = str(uuid.uuid4())
        threading.Thread(target=self.run, args=(job_id, dataset, epochs, actor), daemon=True).start()
        return job_id

    def run(self, job_id, dataset, epochs, actor):
        directory = settings.data_dir / "runs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with transaction() as session:
                session.add(
                    Record(
                        id=job_id,
                        kind="experiment",
                        payload={"state": "running", "dataset": dataset, "started": time.time()},
                    )
                )
                audit(session, actor, "training_started", job_id, dataset=dataset)
            manifest = settings.data_dir / "datasets" / dataset / "manifest.json"
            with (directory / "process.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "training.pipeline",
                        "train",
                        str(manifest),
                        str(directory),
                        "--epochs",
                        str(epochs),
                    ],
                    stdout=log,
                    stderr=log,
                    timeout=6 * 3600,
                )
            if completed.returncode:
                raise RuntimeError("Training failed; inspect the run process.log")
            result = json.loads((directory / "manifest.json").read_text())
            with transaction() as session:
                record = session.get(Record, job_id)
                record.payload = {"state": "completed", "dataset": dataset, "result": result}
                session.add(ModelVersion(name=result["name"], version=job_id, manifest=result))
                audit(session, actor, "training_calibration_completed", job_id)
        except Exception as exc:
            with transaction() as session:
                record = session.get(Record, job_id)
                if record:
                    record.payload = {"state": "failed", "error": str(exc)}
        finally:
            with self.lock:
                self.running = False


jobs = TrainingJobs()
