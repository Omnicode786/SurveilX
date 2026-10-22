"""Read-only generation progress for detection, event and accuracy runners."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import psutil


def yolo_epoch(row):
    """Normalize only recorded CSV metrics; do not fabricate absent objectives."""
    row = {key.strip(): value for key, value in row.items() if isinstance(key, str)}
    def number(key):
        try:
            result = float(row[key])
            return result if math.isfinite(result) else None
        except (KeyError, TypeError, ValueError):
            return None
    epoch = number("epoch")
    if epoch is None or epoch < 1 or not epoch.is_integer():
        return None
    precision, recall = number("metrics/precision(B)"), number("metrics/recall(B)")
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"epoch": int(epoch), "metric_source": "Ultralytics results.csv validation",
            "validation_map50": number("metrics/mAP50(B)"),
            "validation_metrics": {"precision": precision, "recall": recall, "f1": f1},
            "f1_source": "derived from recorded precision and recall",
            "learning_rate": number("lr/pg0"), "elapsed_seconds": number("time"),
            "training": {"losses": {"box_regression": number("train/box_loss"),
                "classification": number("train/cls_loss"), "distribution_focal": number("train/dfl_loss")}},
            "validation": {"losses": {"box_regression": number("val/box_loss"),
                "classification": number("val/cls_loss"), "distribution_focal": number("val/dfl_loss")}}}


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
        latest_epoch = None
        checkpoint_selection = None
        if re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", version):
            directory = data_dir / "runs" / version
            try:
                history_path = directory / "history.json"
                if not history_path.exists():
                    history_path = directory / "native_history.json"
                if history_path.exists():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    completed_epochs = len(history)
                    latest_epoch = history[-1] if history else None
                elif (directory / "fit/results.csv").exists():
                    with (directory / "fit/results.csv").open() as stream:
                        for row in csv.DictReader(stream):
                            normalized = yolo_epoch(row)
                            if normalized:
                                completed_epochs += 1
                                latest_epoch = normalized
            except (ValueError, OSError):
                pass  # Progress may be in the middle of an epoch write.
            try:
                artifact = json.loads((directory / "manifest.json").read_text())
                checkpoint_selection = {"selected": artifact.get("validation_selection", {}).get("selected"),
                    "best_epoch": artifact.get("best_epoch"), "patience": artifact.get("config", {}).get("patience"),
                    "stopping_reason": artifact.get("stopping", {}).get("stopping_reason")}
            except (ValueError, OSError):
                pass
        job_state = job.get("state", "unknown")
        if job_state == "running" and state in ("interrupted", "process_unverified"):
            job_state = state
        jobs.append({"architecture": plan.get("pipeline", key), "version": version, "state": job_state,
                     "model_id": job.get("model_id"), "completed_epochs": completed_epochs,
                     "latest_epoch": latest_epoch,
                     "checkpoint_selection": checkpoint_selection,
                     "epochs": plan.get("options", {}).get("epochs", journal.get("config", {}).get("epochs")),
                     "comparison": job.get("comparison"), "coverage": job.get("coverage", []),
                     "profiles": job.get("profiles", []),
                     "target": job.get("target", {"minimum": target} if target is not None else None),
                     "error": job.get("error")})
    return {"generation": path.parent.name, "state": state, "config": journal.get("config", {}),
            "jobs": jobs, "error": journal.get("error")}
