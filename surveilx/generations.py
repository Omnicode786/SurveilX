"""Read-only generation progress for detection, event and accuracy runners."""

import csv
import hashlib
import json
import re
from pathlib import Path

import psutil


def runner_matches(process, directory):
    command = process.cmdline()
    if "-m" not in command:
        return False
    args = command[command.index("-m") + 1:]
    if len(args) >= 3 and args[0] in ("scripts.train_domain_generation", "scripts.train_event_generation"):
        return args[2] == directory.name
    if len(args) >= 2 and args[0] == "scripts.train_accuracy_campaign":
        plan = Path(args[1])
        if not plan.is_absolute():
            plan = Path(process.cwd()) / plan
        return plan.resolve() == (directory / "plan.json").resolve()
    return False


def queue_matches(process, directory):
    command = process.cmdline()
    if "-m" not in command:
        return False
    args = command[command.index("-m") + 1:]
    if len(args) < 2 or args[0] not in {
        "scripts.queue_accuracy_retry",
        "scripts.queue_ppe_accuracy",
        "scripts.queue_model_optimization",
        "scripts.queue_event_generation",
    }:
        return False
    target = Path(args[1])
    if not target.is_absolute():
        target = Path(process.cwd()) / target
    return target.resolve() == directory.resolve()


def read_queue(path):
    journal = json.loads(path.read_text(encoding="utf-8"))
    state = journal.get("state", "unknown")
    if state in {"waiting", "training", "optimizing"}:
        try:
            if not queue_matches(psutil.Process(journal["pid"]), path.parent):
                state = "interrupted"
        except (psutil.NoSuchProcess, KeyError):
            state = "interrupted"
        except psutil.AccessDenied:
            state = "process_unverified"
    config = {
        key: journal[key]
        for key in (
            "dataset",
            "after_campaign",
            "dataset_ready",
            "predecessor_state",
            "accuracy_g3_state",
            "ppe_state",
            "ppe_retry_state",
        )
        if key in journal
    }
    return {
        "generation": path.parent.name,
        "state": state,
        "config": config,
        "jobs": [],
        "error": journal.get("error"),
    }


def read_generation(path, data_dir):
    journal = json.loads(path.read_text(encoding="utf-8"))
    state = journal.get("state", "unknown")
    if state == "running":
        try:
            if not runner_matches(psutil.Process(journal["pid"]), path.parent):
                state = "interrupted"
        except (psutil.NoSuchProcess, KeyError):
            state = "interrupted"
        except psutil.AccessDenied:
            state = "process_unverified"
    planned, target = {}, None
    if journal.get("plan_sha256"):
        content = (path.parent / "plan.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != journal["plan_sha256"]:
            raise ValueError("Plan integrity mismatch")
        specification = json.loads(content)
        planned = {job["version"]: job for job in specification["jobs"]}
        target = specification.get("target")
    entries = dict(journal.get("jobs", {}))
    if not entries and journal.get("version"):
        entries["scene"] = journal
    for version in planned:
        entries.setdefault(version, {"state": "pending"})
    jobs = []
    for key, job in entries.items():
        plan = planned.get(key, {})
        version = job.get("version", plan.get("version", ""))
        completed_epochs = 0
        if re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
            directory = data_dir / "runs" / version
            try:
                if (directory / "history.json").exists():
                    completed_epochs = len(json.loads((directory / "history.json").read_text()))
                elif (directory / "fit/results.csv").exists():
                    with (directory / "fit/results.csv").open() as stream:
                        completed_epochs = sum(1 for row in csv.DictReader(stream) if row.get("epoch"))
            except (ValueError, OSError):
                pass  # Progress may be in the middle of an epoch write.
        job_state = job.get("state", "unknown")
        if job_state == "running" and state in ("interrupted", "process_unverified"):
            job_state = state
        jobs.append({"architecture": plan.get("pipeline", key), "version": version, "state": job_state,
                     "model_id": job.get("model_id"), "completed_epochs": completed_epochs,
                     "epochs": plan.get("options", {}).get("epochs", journal.get("config", {}).get("epochs")),
                     "comparison": job.get("comparison"), "coverage": job.get("coverage", []),
                     "profiles": job.get("profiles", []),
                     "target": job.get("target", {"minimum": target} if target is not None else None),
                     "error": job.get("error")})
    return {"generation": path.parent.name, "state": state, "config": journal.get("config", {}),
            "jobs": jobs, "error": journal.get("error")}
