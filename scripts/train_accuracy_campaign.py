"""Serial, fingerprinted accuracy experiments; no acceptance or activation side effects."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.register_run import register
from scripts.train_domain_generation import save
from surveilx.config import settings
from training.datasets import digest, validate_manifest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paths(job):
    for key in ("version", "dataset", "parent"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", job[key]):
            raise ValueError(f"Invalid {key}")
    return (settings.data_dir / "datasets" / job["dataset"] / "manifest.json",
            settings.data_dir / "runs" / job["parent"], settings.data_dir / "runs" / job["version"])


def verify_job(job):
    dataset, parent, _ = paths(job)
    metadata = read(parent / "manifest.json")
    if digest(dataset) != job["dataset_sha256"]:
        raise ValueError("Dataset changed since experiment planning")
    if digest(parent / "weights.pt") != job["parent_sha256"] or metadata["weights_sha256"] != job["parent_sha256"]:
        raise ValueError("Parent weights changed since experiment planning")
    manifest, counts = validate_manifest(dataset)
    if metadata["dataset_sha256"] != job["dataset_sha256"]:
        raise ValueError("Paired accuracy comparison requires the parent's frozen dataset")
    if metadata["classes"] != manifest["classes"] or metadata["domain"] != manifest["domain"]:
        raise ValueError("Parent taxonomy/domain differs from dataset")
    return manifest, metadata, counts


def command_for(job):
    dataset, parent, output = paths(job)
    modules = {"scratch": "training.detection_pipeline", "yolo": "training.yolo_pipeline",
               "event": "training.pipeline"}
    command = [sys.executable, "-m", modules[job["pipeline"]], "train", str(dataset), str(output),
               "--initialize-from", str(parent)]
    allowed = {"epochs", "threads", "learning-rate", "patience", "balanced-sampling", "finetune-all",
               "augment", "baseline", "seed"}
    for key, value in job["options"].items():
        if key not in allowed:
            raise ValueError(f"Unknown training option {key}")
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def comparison(parent, candidate):
    task = candidate["task"]
    metric = "map50" if task == "detection" else "accuracy"
    before, after = parent["metrics"][metric], candidate["metrics"][metric]
    classes = []
    old_classes = parent["metrics"].get("per_class", [])
    new_classes = candidate["metrics"].get("per_class", [])
    class_metric = "ap50" if task == "detection" else "recall"
    for index, label in enumerate(candidate["classes"]):
        old = next((item for item in old_classes if item.get("class_id") == index), {})
        new = next((item for item in new_classes if item.get("class_id") == index), {})
        old_value, new_value = old.get(class_metric), new.get(class_metric)
        classes.append({"class": label, "metric": class_metric, "before": old_value, "after": new_value,
                        "delta": new_value - old_value if old_value is not None and new_value is not None else None,
                        "details": new})
    return {"metric": metric, "before": before, "after": after, "delta": after - before,
            "classes": classes, "validation_selection": candidate.get("validation_selection"),
            "test_is_selection_criterion": False, "independent_acceptance": False}


def coverage(manifest):
    result = []
    for index, label in enumerate(manifest["classes"]):
        counts = {}
        for split in ("train", "validation", "calibration", "test"):
            samples = [sample for sample in manifest["samples"] if sample["split"] == split]
            counts[split] = sum(sample.get("labels", []).count(index) if manifest.get("task") == "detection"
                                else int(sample["label"] == index) for sample in samples)
        result.append({"class": label, "labeled_instances": counts})
    return result


def write_report(directory, plan, journal):
    report = {"campaign": plan["name"], "state": journal["state"], "updated": time.time(),
              "scope": "Paired development evaluation on frozen splits; candidates stay inactive",
              "unsupported": plan.get("unsupported", []), "jobs": journal["jobs"]}
    save(directory / "comparison.json", report)
    lines = ["# Accuracy campaign", "", report["scope"], "",
             "| Candidate | State | Metric | Parent | Candidate | Change |",
             "|---|---|---|---:|---:|---:|"]
    for job in plan["jobs"]:
        item = journal["jobs"].get(job["version"], {"state": "pending"})
        metrics = item.get("comparison")
        values = (f"{metrics['metric']} | {metrics['before']:.4f} | {metrics['after']:.4f} | {metrics['delta']:+.4f}"
                  if metrics else "pending | — | — | —")
        lines.append(f"| {job['version']} | {item['state']} | {values} |")
    lines += ["", "Class results and split coverage, including regressions and absent classes, are in comparison.json.",
              "Test scores are reports only; checkpoint selection uses validation. Repeated development tests do not replace independent acceptance.",
              "", "## Remaining data and hardware gaps", ""]
    lines.extend(f"- {reason}" for reason in plan.get("unsupported", []))
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan, directory = read(plan_path), plan_path.parent
    if not plan.get("jobs") or len({job["version"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("Plan needs unique candidate versions")
    journal_path, lock_path = directory / "status.json", directory / "runner.lock"
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as lock:
        lock.write(str(os.getpid()))
    journal = None
    try:
        fingerprint = digest(plan_path)
        journal = read(journal_path) if journal_path.exists() else {"plan_sha256": fingerprint, "jobs": {}}
        if journal["plan_sha256"] != fingerprint:
            journal = None
            raise ValueError("Campaign plan changed; choose a new campaign directory")
        journal.update(state="running", pid=os.getpid(), updated=time.time())
        save(journal_path, journal)
        write_report(directory, plan, journal)
        for job in plan["jobs"]:
            version = job["version"]
            item = journal["jobs"].setdefault(version, {})
            try:
                manifest, parent, counts = verify_job(job)
                _, _, output = paths(job)
                artifact = output / "manifest.json"
                item.update(coverage=coverage(manifest), counts=counts)
                if not artifact.exists():
                    if output.exists() and any(output.iterdir()):
                        raise ValueError(f"Partial run preserved at {output}; inspect it before creating a new version")
                    command = command_for(job)
                    log_path = directory / f"{version}.log"
                    item.update(state="running", command=command, log=str(log_path), started=time.time())
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        item["pid"] = process.pid
                        save(journal_path, journal)
                        write_report(directory, plan, journal)
                        code = process.wait()
                    if code or not artifact.exists():
                        raise RuntimeError(f"Training failed with code {code}; see {log_path}")
                candidate = read(artifact)
                initialization = candidate.get("initialization", candidate.get("config", {}).get("initialization"))
                if (candidate["dataset_sha256"] != job["dataset_sha256"] or initialization != job["parent_sha256"]
                        or digest(output / "weights.pt") != candidate["weights_sha256"]):
                    raise ValueError("Candidate dataset, parent lineage or checkpoint integrity mismatch")
                item.update(state="completed", model_id=register(version), comparison=comparison(parent, candidate),
                            finished=time.time())
                item.pop("error", None)
            except Exception as exc:
                item.update(state="failed", error=str(exc), finished=time.time())
            save(journal_path, journal)
            write_report(directory, plan, journal)
        journal.update(state="completed" if all(j["state"] == "completed" for j in journal["jobs"].values())
                       else "completed_with_failures", finished=time.time())
        save(journal_path, journal)
        write_report(directory, plan, journal)
        return journal
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["state"] == "completed" else 1)
