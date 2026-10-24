"""Serial, fingerprinted accuracy experiments; no acceptance or activation side effects."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.register_run import register
from scripts.train_domain_generation import save
from surveilx.config import settings
from surveilx.training_runtime import training_interpreter, available_commit_bytes
from training.datasets import digest, validate_manifest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resource_failure(path):
    """Classify only the current failed child's bounded log tail."""
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        tail = stream.read().decode('utf-8', errors='replace')
    markers = ('Training resource guard:', 'DefaultCPUAllocator: not enough memory',
        'CUDA out of memory', 'CUDA error: out of memory', 'No space left on device')
    for line in reversed(tail.splitlines()):
        if any(marker in line for marker in markers):
            return line.strip()[-1000:]
    return None


def paths(job):
    for key in ("version", "dataset", "parent"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", job[key]):
            raise ValueError(f"Invalid {key}")
    return (
        settings.data_dir / "datasets" / job["dataset"] / "manifest.json",
        settings.data_dir / "runs" / job["parent"],
        settings.data_dir / "runs" / job["version"],
    )


def verify_expansion(previous, expanded):
    """Existing source groups/content must retain their train/holdout role."""
    groups, hashes = {}, {}
    for sample in previous["samples"]:
        groups[sample["group"]] = sample["split"]
        for key in ("sha256", "source_sha256"):
            if sample.get(key):
                hashes[sample[key]] = sample["split"]
    for sample in expanded["samples"]:
        if sample["group"] in groups and groups[sample["group"]] != sample["split"]:
            raise ValueError("Expanded dataset moved an existing source group between splits")
        for key in ("sha256", "source_sha256"):
            value = sample.get(key)
            if value in hashes and hashes[value] != sample["split"]:
                raise ValueError("Expanded dataset moved existing source content between splits")


def verify_job(job):
    for relative, fingerprint in job.get("code_sha256", {}).items():
        source = (Path.cwd() / relative).resolve()
        if not source.is_relative_to(Path.cwd().resolve()) or digest(source) != fingerprint:
            raise ValueError("Training source changed since experiment planning")
    dataset, parent, _ = paths(job)
    metadata = read(parent / "manifest.json")
    if digest(dataset) != job["dataset_sha256"]:
        raise ValueError("Dataset changed since experiment planning")
    if (
        digest(parent / "weights.pt") != job["parent_sha256"]
        or metadata["weights_sha256"] != job["parent_sha256"]
    ):
        raise ValueError("Parent weights changed since experiment planning")
    manifest, counts = validate_manifest(dataset)
    expanded = metadata["dataset_sha256"] != job["dataset_sha256"]
    if expanded:
        previous_name = job.get("parent_dataset", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", previous_name):
            raise ValueError("Changed data requires an explicit parent_dataset for re-evaluation")
        previous_path = settings.data_dir / "datasets" / previous_name / "manifest.json"
        if digest(previous_path) != metadata["dataset_sha256"]:
            raise ValueError("Parent dataset does not match model provenance")
        previous, _ = validate_manifest(previous_path)
        verify_expansion(previous, manifest)
    if metadata["classes"] != manifest["classes"] or metadata["domain"] != manifest["domain"]:
        raise ValueError("Parent taxonomy/domain differs from dataset")
    if job["pipeline"] == "yolo" and job["options"].get("initialize-partial"):
        from training.yolo_initialization import verify_partial
        options = job["options"]
        verify_partial(options["initialize-partial"], options["partial-sha256"], options["partial-plan"],
            options["partial-plan-sha256"], job["dataset_sha256"], job["parent_sha256"])
    if expanded:
        if manifest.get("task") != "detection":
            raise ValueError("Expanded campaign comparison currently supports detection only")
        from training.acceptance import predict_artifact

        # Use only the frozen model and its existing calibration. This is a development
        # comparison; independent acceptance is not requested or granted by this evaluator.
        _, measured = predict_artifact(parent, dataset, manifest)
        metadata["metrics"] = measured
        metadata["comparison_dataset_sha256"] = job["dataset_sha256"]
    if metadata.get("task", metadata.get("calibration_task")) == "event" and not metadata["metrics"].get(
        "per_class"
    ):
        from training.calibration import metrics

        # Older manifests predate per-class metrics. Derive them from their saved test outputs,
        # verifying labels and the original aggregate score rather than inventing missing scores.
        with np.load(parent / "heldout_predictions.npz", allow_pickle=False) as saved:
            expected = np.array(
                [sample["label"] for sample in manifest["samples"] if sample["split"] == "test"]
            )
            if not np.array_equal(saved["labels"], expected):
                raise ValueError("Saved parent test labels do not match the frozen dataset")
            report = metrics(saved["logits"], saved["labels"], metadata["temperature"])
        if not np.isclose(report["accuracy"], metadata["metrics"]["accuracy"]):
            raise ValueError("Saved parent predictions disagree with its recorded test score")
        metadata["metrics"] = report
    return manifest, metadata, counts


def command_for(job, resume_checkpoint=None):
    dataset, parent, output = paths(job)
    modules = {
        "scratch": "training.detection_pipeline",
        "yolo": "training.yolo_pipeline",
        "event": "training.pipeline",
    }
    command = [
        training_interpreter(),
        "-m",
        modules[job["pipeline"]],
        "train",
        str(dataset),
        str(output),
        "--initialize-from",
        str(parent),
    ]
    allowed = {
        "epochs",
        "threads",
        "learning-rate",
        "patience",
        "balanced-sampling",
        "finetune-all",
        "augment",
        "baseline",
        "seed",
        "image-size",
        "imgsz",
        "batch-size",
        "batch",
        "classification-weight",
        "classification-focal-gamma",
        "classification-focal-alpha",
        "architecture-version",
        "trainable-scope",
        "upgrade-from-v1",
        "transfer-from-legacy",
        "assignment",
        "preprocessing",
        "accumulation-steps",
        "initialize-partial",
        "partial-sha256",
        "partial-dataset-sha256",
        "augmentation",
        "training-profile",
        "effective-batch-size",
        "training-decode-mode",
        "adapter-version",
        "class-weights-power",
        "partial-plan",
        "partial-plan-sha256",
    }
    for key, value in job["options"].items():
        if key not in allowed:
            raise ValueError(f"Unknown training option {key}")
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    if resume_checkpoint:
        if job["pipeline"] != "yolo":
            raise ValueError("Only adapted-YOLO runs have a verified interrupted-run checkpoint")
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return command


def comparison(parent, candidate):
    task = candidate["task"]
    metric = "map50" if task == "detection" else "accuracy"
    parent_metrics = parent["metrics"]
    aligned = candidate.get("parent_evaluation")
    if aligned:
        if (aligned.get("dataset_sha256") != candidate["dataset_sha256"]
                or aligned.get("weights_sha256") != parent["weights_sha256"]
                or aligned.get("image_size") != candidate.get("config", {}).get("image_size")
                or aligned.get("preprocessing") != candidate.get("config", {}).get("preprocessing")):
            raise ValueError("Paired parent evaluation contract mismatch")
        parent_metrics = aligned["metrics"]
    before, after = parent_metrics[metric], candidate["metrics"][metric]
    classes = []
    old_classes = parent_metrics.get("per_class", [])
    new_classes = candidate["metrics"].get("per_class", [])
    class_metric = "ap50" if task == "detection" else "recall"
    for index, label in enumerate(candidate["classes"]):
        old = next((item for item in old_classes if item.get("class_id") == index), {})
        new = next((item for item in new_classes if item.get("class_id") == index), {})
        old_value, new_value = old.get(class_metric), new.get(class_metric)
        classes.append(
            {
                "class": label,
                "metric": class_metric,
                "before": old_value,
                "after": new_value,
                "delta": new_value - old_value if old_value is not None and new_value is not None else None,
                "details": new,
            }
        )
    return {
        "metric": metric,
        "before": before,
        "after": after,
        "delta": after - before,
        "classes": classes,
        "validation_selection": candidate.get("validation_selection"),
        "test_is_selection_criterion": False,
        "independent_acceptance": False,
        "evaluation_dataset_sha256": candidate["dataset_sha256"],
        "parent_re_evaluated": bool(aligned) or "comparison_dataset_sha256" in parent,
        "paired_profile": {key: aligned[key] for key in ("image_size", "preprocessing")} if aligned else None,
    }


def verify_recipe(job, candidate):
    pipeline = job["pipeline"]
    expected = "event" if pipeline == "event" else "detection"
    if candidate.get("task") != expected:
        raise ValueError("Candidate task differs from the planned pipeline")
    if pipeline == "scratch" and candidate.get("architecture") != "sva-detector":
        raise ValueError("Candidate architecture differs from the planned scratch pipeline")
    if pipeline == "yolo":
        expected = "yolo_baseline" if job["options"].get("baseline") else "yolo_rai"
        if candidate.get("architecture") != expected:
            raise ValueError("Candidate YOLO architecture differs from the plan")
    config = (
        candidate.get("config", {})
        if pipeline != "event"
        else {
            **candidate.get("training", {}),
            "epochs": candidate.get("epochs"),
            "seed": candidate.get("seed"),
        }
    )
    for key, value in job["options"].items():
        field = {"imgsz": "image_size", "batch": "batch_size"}.get(key, key.replace("-", "_"))
        if key != "baseline" and config.get(field) != value:
            raise ValueError(f"Candidate training recipe differs from plan: {key}")


def coverage(manifest):
    result = []
    for index, label in enumerate(manifest["classes"]):
        counts = {}
        for split in ("train", "validation", "calibration", "test"):
            samples = [sample for sample in manifest["samples"] if sample["split"] == split]
            counts[split] = sum(
                sample.get("labels", []).count(index)
                if manifest.get("task") == "detection"
                else int(sample["label"] == index)
                for sample in samples
            )
        result.append({"class": label, "labeled_instances": counts})
    return result


def write_report(directory, plan, journal):
    report = {
        "campaign": plan["name"],
        "state": journal["state"],
        "updated": time.time(),
        "scope": "Paired development evaluation on frozen splits; candidates stay inactive",
        "unsupported": plan.get("unsupported", []),
        "jobs": journal["jobs"],
    }
    save(directory / "comparison.json", report)
    lines = [
        "# Accuracy campaign",
        "",
        report["scope"],
        "",
        "| Candidate | State | Metric | Parent | Candidate | Change |",
        "|---|---|---|---:|---:|---:|",
    ]
    for job in plan["jobs"]:
        item = journal["jobs"].get(job["version"], {"state": "pending"})
        metrics = item.get("comparison")
        values = (
            f"{metrics['metric']} | {metrics['before']:.4f} | {metrics['after']:.4f} | {metrics['delta']:+.4f}"
            if metrics
            else "pending | — | — | —"
        )
        lines.append(f"| {job['version']} | {item['state']} | {values} |")
    lines += [
        "",
        "The accompanying JSON report records class results and split coverage, including regressions and absent classes.",
        "Test scores are reports only; checkpoint selection uses validation. Repeated development tests do not replace independent acceptance.",
        "",
        "## Remaining data and hardware gaps",
        "",
    ]
    lines.extend(f"- {reason}" for reason in plan.get("unsupported", []))
    lines += ["", "## Class-level results", ""]
    for version, item in journal["jobs"].items():
        result = item.get("comparison")
        if not result:
            continue
        lines += [
            f"### {version}",
            "",
            "| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in result["classes"]:
            counts = next(
                (
                    entry["labeled_instances"]
                    for entry in item.get("coverage", [])
                    if entry["class"] == row["class"]
                ),
                {},
            )
            values = [
                f"{row[key]:.4f}" if row[key] is not None else "not evaluated"
                for key in ("before", "after", "delta")
            ]
            lines.append(
                f"| {row['class']} | {row['metric']} | "
                + " | ".join(values)
                + f" | {counts.get('train', '—')} | {counts.get('test', '—')} |"
            )
        lines.append("")
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
        if journal.get("error"):
            journal.setdefault("historical_errors", []).append({"state": journal.get("state"),
                "error": journal.pop("error"), "updated": journal.get("updated")})
        journal.update(state="running", pid=os.getpid(), updated=time.time())
        save(journal_path, journal)
        write_report(directory, plan, journal)
        for job in plan["jobs"]:
            version = job["version"]
            item = journal["jobs"].setdefault(version, {})
            try:
                if digest(plan_path) != fingerprint:
                    raise ValueError("Campaign plan changed while running; no further training permitted")
                reserve = plan.get("minimum_free_disk_bytes", 0)
                resource_error = None
                if reserve and shutil.disk_usage(directory).free < reserve:
                    resource_error = "Campaign disk reserve reached; preserve artifacts and free space before resuming"
                if plan.get("minimum_available_ram_bytes"):
                    import psutil
                    if psutil.virtual_memory().available < plan["minimum_available_ram_bytes"]:
                        resource_error = "Available RAM below campaign reserve; no next job launched"
                if plan.get("minimum_available_commit_bytes"):
                    committed = available_commit_bytes()
                    if committed is not None and committed < plan["minimum_available_commit_bytes"]:
                        resource_error = "Available Windows commit below campaign reserve; no next job launched"
                if plan.get("minimum_free_gpu_mib"):
                    try:
                        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=5)
                        if float(free.strip().splitlines()[0]) < plan["minimum_free_gpu_mib"]:
                            resource_error = "Free VRAM below campaign reserve; no next job launched"
                    except (OSError, ValueError, subprocess.SubprocessError):
                        resource_error = "Cannot verify free GPU memory; no next job launched"
                if resource_error:
                    item.update(version=version, state="not_started_resource_limit", error=resource_error)
                    journal.update(state="resource_limited", error=resource_error, updated=time.time())
                    save(journal_path, journal)
                    write_report(directory, plan, journal)
                    return journal
                manifest, parent, counts = verify_job(job)
                _, _, output = paths(job)
                artifact = output / "manifest.json"
                item.update(version=version, coverage=coverage(manifest), counts=counts)
                if not artifact.exists():
                    resume_checkpoint = output / "fit" / "weights" / "last.pt"
                    resumable = job["pipeline"] == "yolo" and resume_checkpoint.is_file()
                    if output.exists() and any(output.iterdir()) and not resumable:
                        raise ValueError(
                            f"Partial run preserved at {output}; inspect it before creating a new version"
                        )
                    command = command_for(job, resume_checkpoint if resumable else None)
                    log_path = directory / f"{version}.log"
                    item.update(state="running", command=command, log=str(log_path), resumed=resumable)
                    item["resumed_at" if resumable else "started"] = time.time()
                    with log_path.open("a" if resumable else "w", encoding="utf-8") as log:
                        environment = os.environ.copy()
                        if plan.get("required_cuda"):
                            environment["SURVEILX_REQUIRE_CUDA"] = "1"
                        if plan.get("gpu_memory_fraction") is not None:
                            environment["SURVEILX_GPU_MEMORY_FRACTION"] = str(plan["gpu_memory_fraction"])
                        for key, variable in (("training_min_free_disk_bytes", "SURVEILX_TRAINING_MIN_DISK_BYTES"),
                                ("training_min_available_commit_bytes", "SURVEILX_TRAINING_MIN_COMMIT_BYTES")):
                            if plan.get(key) is not None:
                                environment[variable] = str(plan[key])
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                        item["pid"] = process.pid
                        save(journal_path, journal)
                        write_report(directory, plan, journal)
                        code = process.wait()
                    if code or not artifact.exists():
                        exhausted = resource_failure(log_path)
                        if exhausted:
                            item.update(state="failed_resource_limit", error=exhausted, finished=time.time())
                            journal.update(state="resource_limited", error=exhausted, updated=time.time())
                            save(journal_path, journal)
                            write_report(directory, plan, journal)
                            return journal
                        raise RuntimeError(f"Training failed with code {code}; see {log_path}")
                candidate = read(artifact)
                if plan.get("required_cuda") and candidate.get("device") != "cuda":
                    raise ValueError("CUDA campaign candidate was not trained on CUDA")
                verify_recipe(job, candidate)
                initialization = candidate.get(
                    "initialization", candidate.get("config", {}).get("initialization")
                )
                if (
                    candidate["dataset_sha256"] != job["dataset_sha256"]
                    or initialization != job["parent_sha256"]
                    or digest(output / "weights.pt") != candidate["weights_sha256"]
                ):
                    raise ValueError("Candidate dataset, parent lineage or checkpoint integrity mismatch")
                item.update(
                    state="completed",
                    model_id=register(version),
                    comparison=comparison(parent, candidate),
                    finished=time.time(),
                )
                if "target" in plan:
                    score = item["comparison"]["after"]
                    item["target"] = {
                        "minimum": plan["target"],
                        "observed": score,
                        "met": score >= plan["target"],
                        "scope": "aggregate development metric",
                    }
                item.pop("error", None)
            except Exception as exc:
                item.update(state="failed", error=str(exc), finished=time.time())
                item.pop("comparison", None)
                item.pop("target", None)
            save(journal_path, journal)
            write_report(directory, plan, journal)
        journal.update(
            state="completed"
            if all(j["state"] == "completed" for j in journal["jobs"].values())
            else "completed_with_failures",
            finished=time.time(),
        )
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
