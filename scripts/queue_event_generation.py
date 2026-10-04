"""Run a frozen event recipe only after its named predecessor queue finishes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_event_generation import save
from surveilx.config import settings
from training.datasets import digest, validate_manifest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_plan(plan):
    dataset = settings.data_dir / "datasets" / plan["dataset"] / "manifest.json"
    if not dataset.resolve().is_relative_to((settings.data_dir / "datasets").resolve()):
        raise ValueError("Dataset path leaves the dataset root")
    if digest(dataset) != plan["dataset_sha256"]:
        raise ValueError("Queued event dataset changed")
    validate_manifest(dataset)
    for name, checksum in plan.get("code_sha256", {}).items():
        source = Path(name).resolve()
        if not source.is_relative_to(Path.cwd()) or digest(source) != checksum:
            raise ValueError("Queued training code changed; review and freeze a new plan")


def queue(directory):
    directory = Path(directory).resolve()
    plan_path = directory / "event-plan.json"
    plan, checksum = read(plan_path), digest(plan_path)
    if directory.name != plan["generation"]:
        raise ValueError("Generation directory and plan do not match")
    verify_plan(plan)
    lock = directory / "queue.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    journal_path = directory / "queue.json"
    state = {
        "state": "waiting",
        "pid": os.getpid(),
        "started": time.time(),
        "dataset": plan["dataset"],
        "after_campaign": str(plan["after"]),
        "plan_sha256": checksum,
    }
    try:
        previous = read(journal_path) if journal_path.exists() else {}
        if previous.get("plan_sha256", checksum) != checksum:
            raise ValueError("Queued plan changed; preserve the generation")
        while True:
            after = Path(plan["after"])
            predecessor = read(after).get("state") if after.exists() else "pending"
            state.update(predecessor_state=predecessor, updated=time.time())
            save(journal_path, state)
            if predecessor in {"completed", "completed_with_failures"}:
                break
            if predecessor in {"failed", "interrupted"}:
                raise RuntimeError("Predecessor failed; inspect before starting another CPU job")
            time.sleep(30)
        if digest(plan_path) != checksum:
            raise ValueError("Queued plan was modified while waiting")
        verify_plan(plan)
        command = [
            sys.executable,
            "-m",
            "scripts.train_event_generation",
            plan["dataset"],
            plan["generation"],
        ]
        for name in ("epochs", "seed", "threads", "learning_rate", "patience", "batch_size"):
            if plan.get(name) is not None:
                command.extend(["--" + name.replace("_", "-"), str(plan[name])])
        if plan.get("augment"):
            command.append("--augment")
        state.update(state="training", command=command, updated=time.time())
        save(journal_path, state)
        with (directory / "runner.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        outcome = read(directory / "status.json") if (directory / "status.json").exists() else {}
        if result.returncode or outcome.get("state") != "completed":
            raise RuntimeError(f"Event runner failed ({result.returncode}); inspect runner.log")
        state.update(state="completed", finished=time.time())
        save(journal_path, state)
        return state
    except Exception as exc:
        state.update(state="failed", error=str(exc), updated=time.time())
        save(journal_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(queue(parser.parse_args().directory), indent=2))
