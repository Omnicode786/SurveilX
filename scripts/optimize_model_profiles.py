"""Build calibrated detector profiles selected by validation AP50 and measured latency."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.register_run import register
from scripts.train_domain_generation import save
from surveilx.config import settings
from surveilx.model_profiles import select_tier_profiles
from training.datasets import digest, validate_manifest
from training.detection_pipeline import (
    DetectionImages,
    calibrated_predictions,
    collate_detection,
    detection_reliability,
    evaluate_detections,
    fit_detection_calibration,
    predict,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_yolo_latency(rows):
    values = [sum(float(row.get(key, 0)) for key in ("preprocess", "inference", "postprocess")) for row in rows]
    return float(np.mean(values))


def predictions(expert, metadata, dataset_path, dataset, split, image_size, threads):
    torch.set_num_threads(threads)
    if metadata["architecture"] == "sva-detector":
        loader = DataLoader(
            DetectionImages(
                dataset_path.parent,
                [sample for sample in dataset["samples"] if sample["split"] == split],
                image_size,
            ),
            batch_size=1,
            collate_fn=collate_detection,
        )
        result, targets, timing = predict(expert.model, loader, expert.device)
        return result, targets, timing["mean_batch_amortized_latency_ms"]
    from training.yolo_pipeline import predict_split

    result, targets, timing = predict_split(
        expert.model, dataset_path, dataset, split, image_size, expert.device
    )
    return result, targets, mean_yolo_latency(timing)


def load_expert(run, architecture):
    if architecture == "sva-detector":
        from surveilx.detector_expert import ScratchDetector

        return ScratchDetector(run)
    if architecture in {"yolo_rai", "yolo_baseline"}:
        from surveilx.detector_expert import AdaptedYOLODetector

        return AdaptedYOLODetector(run)
    raise ValueError("Profile optimization supports trained scratch and adapted/baseline YOLO detectors")


def trial(expert, metadata, dataset_path, dataset, image_size, threads):
    calibration_predictions, calibration_targets, _ = predictions(
        expert, metadata, dataset_path, dataset, "calibration", image_size, threads
    )
    calibration = fit_detection_calibration(calibration_predictions, calibration_targets)
    validation_predictions, validation_targets, latency = predictions(
        expert, metadata, dataset_path, dataset, "validation", image_size, threads
    )
    metrics = evaluate_detections(
        calibrated_predictions(validation_predictions, calibration),
        validation_targets,
        len(dataset["classes"]),
        calibration["threshold"],
    )
    return {
        "image_size": image_size,
        "threads": threads,
        "validation_map50": metrics["map50"],
        "validation_metrics": metrics,
        "latency_ms": latency,
        "latency_scope": "model inference and decode/NMS; excludes file loading",
        "calibration": calibration,
    }


def test_profile(expert, metadata, dataset_path, dataset, profile):
    raw, targets, latency = predictions(
        expert,
        metadata,
        dataset_path,
        dataset,
        "test",
        profile["image_size"],
        profile["threads"],
    )
    calibrated = calibrated_predictions(raw, profile["calibration"])
    metrics = evaluate_detections(
        calibrated, targets, len(dataset["classes"]), profile["calibration"]["threshold"]
    )
    return {
        **profile,
        "test_metrics": metrics,
        "test_reliability": detection_reliability(raw, targets, profile["calibration"]),
        "test_latency_ms": latency,
        "test_is_selection_criterion": False,
    }, raw


def optimize(job):
    source = settings.data_dir / "runs" / job["source"]
    dataset_path = settings.data_dir / "datasets" / job["dataset"] / "manifest.json"
    output = settings.data_dir / "runs" / job["version"]
    metadata = read(source / "manifest.json")
    dataset, counts = validate_manifest(dataset_path)
    if output.exists():
        artifact = output / "manifest.json"
        if not artifact.exists():
            raise ValueError(f"Partial optimized artifact is preserved at {output}")
        result = read(artifact)
        if result.get("optimization", {}).get("plan_job") != job:
            raise ValueError("Existing optimized artifact belongs to another job")
        return register(job["version"]), result
    if (
        digest(source / "weights.pt") != job["source_weights_sha256"]
        or metadata["weights_sha256"] != job["source_weights_sha256"]
        or digest(dataset_path) != job["dataset_sha256"]
        or metadata["dataset_sha256"] != job["dataset_sha256"]
    ):
        raise ValueError("Source model or dataset changed after optimization planning")
    if metadata.get("task") != "detection" or metadata["classes"] != dataset["classes"]:
        raise ValueError("Optimization job requires a matching detection dataset")
    expert = load_expert(source, metadata["architecture"])
    trials = [
        trial(expert, metadata, dataset_path, dataset, size, job.get("threads", 3))
        for size in job["resolutions"]
    ]
    selected = select_tier_profiles(trials)
    tested, serialized_predictions = {}, {}
    for tier, profile in selected.items():
        key = (profile["image_size"], profile["threads"])
        if key not in tested:
            tested[key] = test_profile(expert, metadata, dataset_path, dataset, profile)
        measured, raw = tested[key]
        selected[tier] = {"tier": tier, **measured}
        serialized_predictions[tier] = [{k: v.tolist() for k, v in row.items()} for row in raw]
    performance = selected["performance"]
    with tempfile.TemporaryDirectory(prefix=".profiles-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        shutil.copy2(source / "weights.pt", staged / "weights.pt")
        for optional in ("dataset_hashes.json",):
            if (source / optional).is_file():
                shutil.copy2(source / optional, staged / optional)
        (staged / "test_predictions.json").write_text(
            json.dumps(serialized_predictions["performance"], indent=2), encoding="utf-8"
        )
        artifact = {
            **metadata,
            "name": f"{metadata['name']} optimized profiles",
            "stage": "candidate",
            "dataset_counts": counts,
            "training_config": metadata.get("training_config", metadata.get("config", {})),
            "config": {
                **metadata.get("config", {}),
                "image_size": performance["image_size"],
                "runtime_threads": performance["threads"],
            },
            "calibration": performance["calibration"],
            "calibrated": True,
            "metrics": performance["test_metrics"],
            "calibrated_test_reliability": performance["test_reliability"],
            "inference_profiles": list(selected.values()),
            "optimization": {
                "source_version": job["source"],
                "source_manifest_sha256": digest(source / "manifest.json"),
                "plan_job": job,
                "selection_split": "validation",
                "calibration_split": "calibration",
                "test_is_selection_criterion": False,
                "trials": trials,
            },
            "deployment_eligible": False,
            "gate_reason": "Optimized inactive candidate; independent acceptance and hardware validation required",
        }
        (staged / "manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        staged.rename(output)
    return register(job["version"]), artifact


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    directory = plan_path.parent
    lock_path, status_path = directory / "runner.lock", directory / "status.json"
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as lock:
        lock.write(str(os.getpid()))
    try:
        fingerprint = digest(plan_path)
        status = read(status_path) if status_path.exists() else {"plan_sha256": fingerprint, "jobs": {}}
        if status["plan_sha256"] != fingerprint:
            raise ValueError("Optimization plan changed; choose a new directory")
        status.update(state="running", pid=os.getpid(), updated=time.time())
        save(status_path, status)
        for job in plan["jobs"]:
            item = status["jobs"].setdefault(job["version"], {})
            try:
                item.update(state="running", started=time.time())
                save(status_path, status)
                model_id, artifact = optimize(job)
                item.update(
                    state="completed",
                    model_id=model_id,
                    source=job["source"],
                    profiles=artifact["inference_profiles"],
                    finished=time.time(),
                )
                item.pop("error", None)
            except Exception as exc:
                item.update(state="failed", error=str(exc), finished=time.time())
            save(status_path, status)
        status.update(
            state="completed"
            if all(item["state"] == "completed" for item in status["jobs"].values())
            else "completed_with_failures",
            finished=time.time(),
        )
        save(status_path, status)
        return status
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    result = run(parser.parse_args().plan)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["state"] == "completed" else 1)
