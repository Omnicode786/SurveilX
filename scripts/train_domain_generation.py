"""Run scratch and adapted-YOLO candidates serially with a durable continuation journal."""

import argparse
import json
import os
import re
import subprocess
import time

from scripts.register_run import register
from surveilx.config import settings
from surveilx.training_runtime import training_interpreter
from training.datasets import digest, validate_manifest


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(dataset, generation, epochs=3, threads=3, batch_size=None):
    if batch_size is not None and not 1 <= batch_size <= 256:
        raise ValueError("Batch size must be 1..256")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) for value in (dataset, generation)):
        raise ValueError("Dataset and generation must be simple directory names")
    manifest_path = settings.data_dir / "datasets" / dataset / "manifest.json"
    manifest, counts = validate_manifest(manifest_path)
    if manifest.get("task") != "detection":
        raise ValueError("This generation runner requires detection data")
    directory = settings.data_dir / "generations" / generation
    directory.mkdir(parents=True, exist_ok=True)
    journal_path, lock_path = directory / "status.json", directory / "runner.lock"
    config = {
        "dataset": dataset,
        "dataset_sha256": digest(manifest_path),
        "epochs": epochs,
        "threads": threads,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size
    # O_EXCL prevents concurrent invocations from training into the same run directories.
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    owns_generation = False
    try:
        journal = (
            json.loads(journal_path.read_text())
            if journal_path.exists()
            else {"config": config, "jobs": {}, "counts": counts, "created": time.time()}
        )
        if journal["config"] != config:
            raise ValueError("Generation configuration changed; use a new generation name")
        owns_generation = True
        journal.update(state="running", pid=os.getpid(), updated=time.time())
        save(journal_path, journal)
        for architecture in ("scratch", "yolo-rai"):
            version = f"{generation}-{architecture}"
            if len(version) > 100:
                raise ValueError("Generated run version is too long")
            output = settings.data_dir / "runs" / version
            artifact = output / "manifest.json"
            if artifact.exists():
                metadata = json.loads(artifact.read_text())
                if metadata.get("dataset_sha256") != config["dataset_sha256"]:
                    raise ValueError("Existing run belongs to another dataset")
                model_id = register(version)
                journal["jobs"][architecture] = {
                    "state": "completed",
                    "version": version,
                    "model_id": model_id,
                }
                save(journal_path, journal)
                continue
            resume_checkpoint = output / "fit" / "weights" / "last.pt"
            resumable = architecture == "yolo-rai" and resume_checkpoint.is_file()
            if output.exists() and any(output.iterdir()) and not resumable:
                raise ValueError(
                    f"Interrupted run preserved at {output}; inspect it and use a new generation name"
                )
            module = "training.detection_pipeline" if architecture == "scratch" else "training.yolo_pipeline"
            command = [
                training_interpreter(),
                "-m",
                module,
                "train",
                str(manifest_path),
                str(output),
                "--epochs",
                str(epochs),
                "--threads",
                str(threads),
            ]
            if resumable:
                command.extend(["--resume-checkpoint", str(resume_checkpoint)])
            if batch_size is not None:
                command.extend(["--batch-size" if architecture == "scratch" else "--batch", str(batch_size)])
            log_path = directory / f"{architecture}.log"
            journal["jobs"][architecture] = {
                "state": "running",
                "version": version,
                "log": str(log_path),
                "command": command,
                "started": time.time(),
                "resumed": resumable,
            }
            save(journal_path, journal)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode or not artifact.exists():
                journal["jobs"][architecture]["state"] = "failed"
                raise RuntimeError(f"{architecture} failed with code {result.returncode}; inspect {log_path}")
            model_id = register(version)
            journal["jobs"][architecture].update(state="completed", model_id=model_id, finished=time.time())
            save(journal_path, journal)
        journal.update(state="completed", finished=time.time())
        save(journal_path, journal)
        return journal
    except Exception as exc:
        if owns_generation:
            journal.update(state="failed", error=str(exc), updated=time.time())
            save(journal_path, journal)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("generation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.threads <= 32:
        parser.error("epochs must be 1..100 and threads 1..32")
    print(json.dumps(run(**vars(args)), indent=2))
