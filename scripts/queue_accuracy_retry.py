"""Run a frozen retry campaign after its predecessor reaches a terminal state."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_accuracy_campaign import read
from scripts.train_domain_generation import save
from training.datasets import digest


TERMINAL_STATES = {"completed", "completed_with_failures"}


def queue(directory, predecessor):
    directory = Path(directory).resolve()
    predecessor = Path(predecessor).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Frozen retry plan is missing: {plan_path}")

    lock = directory / "queue.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    status_path = directory / "queue.json"
    state = {
        "state": "waiting",
        "pid": os.getpid(),
        "started": time.time(),
        "after_campaign": predecessor.parent.name,
        "plan_sha256": digest(plan_path),
    }
    try:
        while True:
            previous = read(predecessor) if predecessor.exists() else {}
            predecessor_state = previous.get("state")
            state.update(state="waiting", predecessor_state=predecessor_state, updated=time.time())
            save(status_path, state)
            if predecessor_state in TERMINAL_STATES:
                break
            if predecessor_state in {"failed", "interrupted"}:
                raise RuntimeError("Predecessor campaign failed; preserve and inspect its journal")
            time.sleep(30)

        if digest(plan_path) != state["plan_sha256"]:
            raise ValueError("Frozen campaign plan changed while waiting; preserve the generation")
        state.update(state="training", updated=time.time())
        save(status_path, state)
        with (directory / "runner.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.train_accuracy_campaign", str(plan_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        campaign = read(directory / "status.json") if (directory / "status.json").exists() else {}
        campaign_state = campaign.get("state")
        if campaign_state not in TERMINAL_STATES:
            raise RuntimeError(f"Retry campaign exited with code {result.returncode}; inspect runner.log")
        state.update(state=campaign_state, finished=time.time())
        save(status_path, state)
        return state
    except Exception as exc:
        state.update(state="failed", error=str(exc), updated=time.time())
        save(status_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("predecessor")
    args = parser.parse_args()
    print(json.dumps(queue(args.directory, args.predecessor), indent=2))
