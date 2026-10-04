"""Queue existing-data detector profile optimization after active accuracy campaigns."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_domain_generation import save
from surveilx.config import settings
from training.datasets import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolution_candidates(image_size):
    if image_size <= 256:
        values = (max(128, image_size - 96), max(160, image_size - 32), image_size)
    elif image_size <= 384:
        values = (image_size - 128, image_size - 64, image_size)
    else:
        values = (image_size - 192, image_size - 96, image_size)
    return sorted({max(128, int(round(value / 32)) * 32) for value in values})


def complete_calibrated_source(version):
    """Accept only a complete, calibrated artifact with its recorded checkpoint."""
    run = settings.data_dir / "runs" / version
    manifest_path, weights_path = run / "manifest.json", run / "weights.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        return False
    try:
        manifest = read(manifest_path)
        return (
            manifest.get("task") == "detection"
            and manifest.get("calibrated") is True
            and manifest.get("calibration", {}).get("status") == "fitted"
            and manifest.get("weights_sha256") == digest(weights_path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def choose_source(preferred, fallback):
    if complete_calibrated_source(preferred):
        return preferred
    if complete_calibrated_source(fallback):
        return fallback
    raise ValueError(f"Neither preferred nor fallback artifact is complete and calibrated: {preferred}, {fallback}")


def retry_state(directory):
    queue_path = directory / "queue.json"
    plan_path = directory / "plan.json"
    if queue_path.exists():
        return read(queue_path).get("state", "unknown")
    if plan_path.exists():
        return "planned"
    return "absent"


def build_plan(directory):
    sources = [
        ("accuracy-g2-person-yolo", "accuracy-g2-person-yolo", "pennfudan", "pennfudan"),
        ("accuracy-g2-person-scratch", "accuracy-g2-person-scratch", "pennfudan", "pennfudan"),
        ("accuracy-g2-person-baseline", "accuracy-g2-person-baseline", "pennfudan", "pennfudan"),
        ("accuracy-g3-weapons-yolo", "accuracy-g2-weapons-yolo", "dangerous-items-development-v1", "dangerous-items-development-v1"),
        ("accuracy-g3b-weapons-scratch", "accuracy-g2-weapons-scratch", "dangerous-items-development-v1", "dangerous-items-development-v1"),
        ("accuracy-g3-fire-yolo", "accuracy-g2-fire-yolo", "dfire-development-v1", "dfire-development-v1"),
        ("accuracy-g3-fire-scratch", "accuracy-g2-fire-scratch", "dfire-development-v1", "dfire-development-v1"),
        ("accuracy-ppe-g3-yolo", "accuracy-g2-ppe-yolo", "sh17-development-v2", "sh17-development-v1"),
        ("accuracy-ppe-g3b-scratch", "accuracy-g2-ppe-scratch", "sh17-development-v2", "sh17-development-v1"),
    ]
    jobs = []
    for preferred, fallback, preferred_dataset, fallback_dataset in sources:
        source = choose_source(preferred, fallback)
        dataset = preferred_dataset if source == preferred else fallback_dataset
        run = settings.data_dir / "runs" / source
        dataset_path = settings.data_dir / "datasets" / dataset / "manifest.json"
        metadata = read(run / "manifest.json")
        if not metadata.get("calibrated"):
            raise ValueError(f"Source model is not calibrated: {source}")
        jobs.append(
            {
                "version": f"{source}-profiles-g1",
                "source": source,
                "dataset": dataset,
                "source_weights_sha256": digest(run / "weights.pt"),
                "dataset_sha256": digest(dataset_path),
                "resolutions": resolution_candidates(metadata["config"]["image_size"]),
                "threads": 3,
            }
        )
    return {
        "name": directory.name,
        "selection": "Calibration per resolution; validation-only accuracy/latency tier selection; test report after selection",
        "jobs": jobs,
        "fixed_contract_models": ["accuracy-g2-fall", "accuracy-g2-entity-v1", "accuracy-g2-entity-v2"],
        "unsupported": [
            "Scene/entity event input contracts are fixed; arbitrary resolution changes are not valid without retraining.",
            "Inherited general-object candidates lack a matching local labeled domain dataset for accuracy selection.",
            "Profile latency is machine-specific and must be remeasured on each deployment hardware tier.",
        ],
    }


def queue(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "queue.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    status = directory / "queue.json"
    state = {"state": "waiting", "pid": os.getpid(), "started": time.time()}
    try:
        while True:
            g3 = read(settings.data_dir / "generations/accuracy-g3/status.json")
            ppe = read(settings.data_dir / "generations/accuracy-ppe-g3/queue.json")
            ppe_retry = retry_state(settings.data_dir / "generations/accuracy-ppe-g3b-scratch")
            ppe_done = ppe.get("state") in {"completed", "failed"}
            retry_done = ppe_retry in {"absent", "completed", "completed_with_failures", "failed"}
            ready = g3.get("state") in {"completed", "completed_with_failures"} and ppe.get("state") in {
                "completed",
                "failed",
            } and retry_done
            state.update(
                state="waiting",
                accuracy_g3_state=g3.get("state"),
                ppe_state=ppe.get("state"),
                ppe_retry_state=ppe_retry if ppe_done else "blocked",
                updated=time.time(),
            )
            save(status, state)
            if ready:
                break
            time.sleep(30)
        plan_path = directory / "plan.json"
        plan = build_plan(directory)
        if plan_path.exists() and read(plan_path) != plan:
            raise ValueError("Optimization plan changed; preserve the existing generation")
        if not plan_path.exists():
            save(plan_path, plan)
        state.update(state="optimizing", plan_sha256=digest(plan_path), updated=time.time())
        save(status, state)
        with (directory / "runner.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.optimize_model_profiles", str(plan_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        state.update(state="completed" if result.returncode == 0 else "failed", finished=time.time())
        save(status, state)
        return state
    except Exception as exc:
        state.update(state="failed", error=str(exc), updated=time.time())
        save(status, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(queue(parser.parse_args().directory), indent=2))
