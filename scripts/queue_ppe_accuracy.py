"""Wait for the balanced dataset and serial campaign, then train PPE without overlap."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.train_accuracy_campaign import read
from scripts.train_domain_generation import save
from surveilx.config import settings
from training.datasets import digest, validate_manifest


def acquisition_running():
    for process in psutil.process_iter(["cmdline"]):
        command = process.info["cmdline"] or []
        if "-m" in command and "scripts.prepare_sh17_balanced" in command:
            return True
    return False


def build_plan(directory, dataset_path):
    _, counts = validate_manifest(dataset_path)
    jobs = []
    for pipeline in ("yolo", "scratch"):
        parent = f"accuracy-g2-ppe-{pipeline}"
        options = {"epochs": 40, "threads": 3, "learning-rate": 0.001 if pipeline == "yolo" else 0.0005,
                   "patience": 12, "imgsz" if pipeline == "yolo" else "image-size": 512,
                   "batch" if pipeline == "yolo" else "batch-size": 4}
        if pipeline == "scratch":
            options.update({"balanced-sampling": True, "classification-weight": 17.0})
        jobs.append({"version": f"{directory.name}-{pipeline}", "pipeline": pipeline,
                     "dataset": dataset_path.parent.name, "parent_dataset": "sh17-development-v1",
                     "parent": parent, "parent_sha256": digest(settings.data_dir / "runs" / parent / "weights.pt"),
                     "dataset_sha256": digest(dataset_path), "options": options})
    return {"name": directory.name, "target": 0.5, "counts": counts, "jobs": jobs,
            "selection": "Validation-only checkpoint selection; both parents re-evaluated on expanded test data",
            "unsupported": ["Parent and candidate share the expanded test data; old SH17 test scores are not the baseline.",
                            "Aggregate AP50 >=0.50 does not certify every PPE class or correct equipment use.",
                            "Photographer-disjoint development data do not establish independent deployment-site acceptance.",
                            "512-pixel models cost more CPU time and memory than the previous 256-pixel runs."]}


def queue(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "queue.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    status = directory / "queue.json"
    state = {"state": "waiting", "pid": os.getpid(), "started": time.time(), "acquisition_retries": 0,
             "dataset": "sh17-development-v2", "after_campaign": "accuracy-g3b-weapons-scratch"}
    dataset_path = settings.data_dir / "datasets/sh17-development-v2/manifest.json"
    acquisition = dataset_path.parent / "acquisition.json"
    predecessor = settings.data_dir / "generations/accuracy-g3b-weapons-scratch/status.json"
    try:
        while True:
            previous = read(predecessor) if predecessor.exists() else {}
            acquisition_state = read(acquisition) if acquisition.exists() else {}
            ready = dataset_path.exists() and acquisition_state.get("state") == "completed"
            training_finished = previous.get("state") in ("completed", "completed_with_failures")
            state.update(state="waiting", dataset_ready=ready, predecessor_state=previous.get("state"), updated=time.time())
            save(status, state)
            if ready and training_finished:
                break
            if not ready and not acquisition_running():
                if state["acquisition_retries"] >= 3:
                    raise RuntimeError("PPE acquisition failed three retries; inspect preserved acquisition logs")
                state["acquisition_retries"] += 1
                save(status, state)
                with (directory / "acquisition-retries.log").open("a", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-m", "scripts.prepare_sh17_balanced", str(dataset_path.parent)],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                continue
            if previous.get("state") == "running":
                try:
                    command = psutil.Process(previous["pid"]).cmdline()
                    if "scripts.train_accuracy_campaign" not in command:
                        raise RuntimeError("Predecessor PID no longer belongs to the accuracy runner")
                except psutil.NoSuchProcess as exc:
                    raise RuntimeError("Predecessor campaign interrupted; inspect its journal before continuing") from exc
            elif previous.get("state") in ("failed", "interrupted"):
                raise RuntimeError("Predecessor campaign failed; preserve and inspect its journal")
            time.sleep(30)
        plan_path = directory / "plan.json"
        plan = build_plan(directory, dataset_path)
        if plan_path.exists():
            if read(plan_path) != plan:
                raise ValueError("Queued plan changed; preserve its existing artifacts")
        else:
            save(plan_path, plan)
        state.update(state="training", plan_sha256=digest(plan_path), updated=time.time())
        save(status, state)
        # Separate process lets the app verify the standard runner module and plan path.
        with (directory / "runner.log").open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-m", "scripts.train_accuracy_campaign", str(plan_path)],
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
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
