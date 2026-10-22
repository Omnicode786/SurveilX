"""Populate validation recommendations after named training campaigns finish.

This opt-in local worker only executes evaluation, never training or deployment.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def readiness(journals):
    states = [json.loads(path.read_text(encoding="utf-8")).get("state") for path in journals]
    if any(state in {"stopped_by_user", "cancelled", "failed", "interrupted"} for state in states):
        raise RuntimeError("Campaign stopped or failed; inspect before automatic evaluation")
    if not all(state in {"completed", "completed_with_failures"} for state in states):
        return "Waiting for all named campaigns; no evaluation overlaps training"
    for process in psutil.process_iter(["name", "cmdline"]):
        args = process.info["cmdline"] or []
        if (process.info["name"] or "").lower().startswith("python") and any(
            name in args for name in (
                "training.yolo_pipeline", "training.detection_pipeline", "training.pipeline",
                "scripts.train_accuracy_campaign")
        ):
            return "Waiting for training process exit"
    if psutil.virtual_memory().available < 3_500_000_000:
        return "Waiting for 3.5 GB available RAM"
    return None


def finalize(journals, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))

    def save(state, **extra):
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps({"state": state, "pid": os.getpid(), "updated": time.time(),
            "journals": [str(p) for p in journals], **extra}, indent=2), encoding="utf-8")
        temp.replace(output)

    try:
        while reason := readiness(journals):
            save("waiting", reason=reason)
            time.sleep(30)
        command = [sys.executable, "-m", "scripts.evaluate_model_catalog"]
        save("evaluating", command=command)
        with output.with_suffix(".log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        reports = output.parent / "status.json"
        entries = json.loads(reports.read_text(encoding="utf-8")) if reports.exists() else []
        failures = [row for row in entries if row.get("state") != "completed"]
        if result.returncode or not entries or failures:
            raise RuntimeError("Validation evaluation incomplete; inspect catalog status/log")
        save("completed", evaluated_reports=len(entries), deployment_changed=False)
    except Exception as exc:
        save("failed", error=str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/model-evaluations/finalizer.json"))
    args = parser.parse_args()
    finalize(args.journals, args.output)


if __name__ == "__main__":
    main()
