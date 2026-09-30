"""Train and register one scene/entity event candidate with a durable journal."""

import argparse
import json
import os
import re
import subprocess
import sys
import time

from scripts.register_run import register
from surveilx.config import settings
from training.datasets import digest, validate_manifest


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(dataset, generation, epochs=5, seed=42):
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) for value in (dataset, generation)):
        raise ValueError("Dataset and generation must be simple directory names")
    manifest_path = settings.data_dir / "datasets" / dataset / "manifest.json"
    manifest, counts = validate_manifest(manifest_path)
    if manifest.get("task", "event") != "event":
        raise ValueError("This generation runner requires video-event data")
    directory = settings.data_dir / "generations" / generation
    directory.mkdir(parents=True, exist_ok=True)
    journal_path, lock_path = directory / "status.json", directory / "runner.lock"
    config = {
        "dataset": dataset,
        "dataset_sha256": digest(manifest_path),
        "epochs": epochs,
        "seed": seed,
    }
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    owns_generation = False
    try:
        journal = (
            json.loads(journal_path.read_text())
            if journal_path.exists()
            else {"config": config, "counts": counts, "created": time.time()}
        )
        if journal["config"] != config:
            raise ValueError("Generation configuration changed; use a new generation name")
        owns_generation = True
        version = f"{generation}-scene"
        output = settings.data_dir / "runs" / version
        artifact = output / "manifest.json"
        if artifact.exists():
            metadata = json.loads(artifact.read_text())
            if metadata.get("dataset_sha256") != config["dataset_sha256"]:
                raise ValueError("Existing run belongs to another dataset")
            journal.update(state="completed", model_id=register(version), version=version, finished=time.time())
            save(journal_path, journal)
            return journal
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Interrupted run preserved at {output}; inspect it and use a new generation name")
        command = [
            sys.executable,
            "-m",
            "training.pipeline",
            "train",
            str(manifest_path),
            str(output),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
        ]
        log_path = directory / "scene.log"
        journal.update(
            state="running", pid=os.getpid(), version=version, command=command, log=str(log_path), started=time.time()
        )
        save(journal_path, journal)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode or not artifact.exists():
            raise RuntimeError(f"event model failed with code {result.returncode}; inspect {log_path}")
        journal.update(state="completed", model_id=register(version), finished=time.time())
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("epochs must be 1..100")
    print(json.dumps(run(**vars(args)), indent=2))
