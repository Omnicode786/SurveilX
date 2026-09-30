"""Independent, frozen-model acceptance evaluation; never fit on acceptance data."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.datasets import digest, validate_manifest


def wilson_lower(successes, total, z=1.96):
    if total <= 0:
        return 0.0
    p = successes / total
    return (p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (
        1 + z * z / total
    )


def verify_independence(acceptance_path, reference_paths):
    acceptance, _ = validate_manifest(acceptance_path)
    if acceptance.get("provenance", {}).get("source_independence_verified") is False:
        raise ValueError("Development-only source groups cannot establish independent acceptance")
    test = [s for s in acceptance["samples"] if s["split"] == "test"]
    key = "image" if acceptance.get("task") == "detection" else "file"
    groups = {s["group"] for s in test}
    hashes = {digest(Path(acceptance_path).parent / s[key]) for s in test}
    for reference_path in reference_paths:
        reference, _ = validate_manifest(reference_path)
        reference_key = "image" if reference.get("task") == "detection" else "file"
        if groups & {s["group"] for s in reference["samples"]}:
            raise ValueError("Acceptance source groups overlap training or earlier evaluation data")
        if hashes & {digest(Path(reference_path).parent / s[reference_key]) for s in reference["samples"]}:
            raise ValueError("Acceptance content overlaps training or earlier evaluation data")
    return acceptance


def predict_artifact(run, dataset_path, dataset):
    run, dataset_path = Path(run), Path(dataset_path)
    metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if digest(run / "weights.pt") != metadata["weights_sha256"]:
        raise ValueError("Model checkpoint checksum mismatch")
    if (
        metadata.get("task", "event") != dataset.get("task", "event")
        or metadata["classes"] != dataset["classes"]
    ):
        raise ValueError("Acceptance task and class order must match the model")
    if metadata.get("domain") != dataset.get("domain") or metadata.get("synthetic", False) != dataset.get(
        "synthetic", False
    ):
        raise ValueError("Acceptance domain and synthetic scope must match the model")
    if not metadata.get("calibrated"):
        raise ValueError("The candidate has no fitted calibration")
    torch.set_num_threads(3)
    if dataset.get("task") == "detection":
        from training.detection_pipeline import (
            DetectionImages,
            calibrated_predictions,
            collate_detection,
            detection_reliability,
            evaluate_detections,
            predict,
        )

        samples = [s for s in dataset["samples"] if s["split"] == "test"]
        started = time.perf_counter()
        if metadata["architecture"] == "sva-detector":
            from surveilx.detector_expert import ScratchDetector

            expert = ScratchDetector(run)
            loader = DataLoader(
                DetectionImages(dataset_path.parent, samples, metadata["config"]["image_size"]),
                batch_size=1,
                collate_fn=collate_detection,
            )
            predictions, targets, _ = predict(expert.model, loader, expert.device)
        elif metadata["architecture"] in {"yolo_rai", "yolo_baseline"}:
            from surveilx.detector_expert import AdaptedYOLODetector
            from training.yolo_pipeline import predict_split

            expert = AdaptedYOLODetector(run)
            predictions, targets, _ = predict_split(
                expert.model, dataset_path, dataset, "test", metadata["config"]["image_size"], expert.device
            )
        else:
            raise ValueError("Unsupported detector architecture")
        metrics = evaluate_detections(
            calibrated_predictions(predictions, metadata["calibration"]),
            targets,
            len(dataset["classes"]),
            metadata["calibration"]["threshold"],
        )
        reliability = detection_reliability(predictions, targets, metadata["calibration"])
        metrics.update(
            ece=reliability["ece"],
            samples=len(samples),
            recall_lower_95=wilson_lower(metrics["true_positives"], metrics["ground_truth"]),
            precision_lower_95=wilson_lower(
                metrics["true_positives"], metrics["true_positives"] + metrics["false_positives"]
            ),
        )
        metrics["per_class_support"] = [r["ground_truth"] for r in metrics["per_class"]]
    else:
        from surveilx.expert import SVAExpert
        from training.pipeline import Clips, predict
        from training.calibration import metrics as event_metrics, probabilities

        contract = {"frames": 8, "entities": 2, "image_size": 64}
        if metadata.get("input_contract", contract) != dataset.get("input_contract", contract):
            raise ValueError("Acceptance clip contract does not match the model")
        samples = [s for s in dataset["samples"] if s["split"] == "test"]
        expert = SVAExpert(run)
        if metadata.get("representation", "entity_clip") != dataset.get("representation", "entity_clip"):
            raise ValueError("Acceptance event representation differs from model")
        loader = DataLoader(Clips(dataset_path.parent, samples, scene=expert.scene), batch_size=1)
        started = time.perf_counter()
        logits, labels = predict(expert.model, loader, expert.device)
        metrics = event_metrics(logits, labels, metadata["temperature"])
        selected = probabilities(logits, metadata["temperature"]).argmax(1)
        metrics["per_class_support"] = [int(np.sum(labels == c)) for c in range(len(dataset["classes"]))]
        metrics["accuracy_lower_95"] = wilson_lower(int(np.sum(selected == labels)), len(labels))
    metrics["wall_ms_per_sample"] = (time.perf_counter() - started) * 1000 / len(samples)
    return metadata, metrics


def gate(metrics, policy, task, baseline=None):
    failures = []

    def minimum(name, value, floor):
        if value is None or not np.isfinite(value) or value < floor:
            failures.append(f"{name} below required {floor}")

    minimum("samples", metrics["samples"], policy["min_samples"])
    minimum("independent groups", metrics["independent_groups"], policy["min_groups"])
    minimum("per-class support", min(metrics["per_class_support"], default=0), policy["min_per_class"])
    if task == "detection":
        minimum("AP50", metrics["map50"], policy["min_ap50"])
        minimum("recall lower 95% bound", metrics["recall_lower_95"], policy["min_recall_lower"])
        minimum("precision lower 95% bound", metrics["precision_lower_95"], policy["min_precision_lower"])
        score = "map50"
    else:
        minimum("accuracy lower 95% bound", metrics["accuracy_lower_95"], policy["min_accuracy_lower"])
        score = "accuracy"
    if metrics["ece"] is None or not np.isfinite(metrics["ece"]) or metrics["ece"] > policy["max_ece"]:
        failures.append("ECE exceeds maximum or is unavailable")
    if (
        not np.isfinite(metrics["wall_ms_per_sample"])
        or metrics["wall_ms_per_sample"] > policy["max_wall_ms"]
    ):
        failures.append("Measured wall time exceeds the configured per-sample limit")
    if baseline and metrics[score] < baseline[score] - policy["max_regression"]:
        failures.append("Candidate regresses against the baseline on the same acceptance set")
    return {"passed": not failures, "failures": failures}


def evaluate(request_path, output):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(request["dataset"])
    metadata = json.loads((Path(request["run"]) / "manifest.json").read_text())
    artifact_hash = digest(Path(request["run"]) / "manifest.json")
    dataset_hash = digest(dataset_path)
    references = [Path(p) for p in request["references"]]
    if metadata["dataset_sha256"] not in {digest(p) for p in references}:
        raise ValueError("Training dataset reference is missing or does not match the candidate provenance")
    if request.get("baseline"):
        previous = json.loads((Path(request["baseline"]) / "manifest.json").read_text())
        if previous["dataset_sha256"] not in {digest(p) for p in references}:
            raise ValueError("Baseline training dataset reference is missing")
    dataset = verify_independence(dataset_path, references)
    metadata, metrics = predict_artifact(request["run"], dataset_path, dataset)
    metrics["independent_groups"] = len({s["group"] for s in dataset["samples"] if s["split"] == "test"})
    baseline = (
        predict_artifact(request["baseline"], dataset_path, dataset)[1] if request.get("baseline") else None
    )
    decision = gate(metrics, request["policy"], metadata.get("task", "event"), baseline)
    if (
        dataset_hash != digest(dataset_path)
        or artifact_hash != digest(Path(request["run"]) / "manifest.json")
        or metadata["weights_sha256"] != digest(Path(request["run"]) / "weights.pt")
    ):
        raise ValueError("Model or dataset changed during acceptance evaluation")
    report = {
        "schema_version": 1,
        "weights_sha256": metadata["weights_sha256"],
        "artifact_manifest_sha256": artifact_hash,
        "dataset_sha256": dataset_hash,
        "request_sha256": digest(request_path),
        "reference_sha256": [digest(p) for p in references],
        "baseline_weights_sha256": previous["weights_sha256"] if request.get("baseline") else None,
        "policy": request["policy"],
        "metrics": metrics,
        "baseline_metrics": baseline,
        "task": metadata.get("task", "event"),
        "domain": dataset["domain"],
        "synthetic": dataset.get("synthetic", False),
        **decision,
        "deployment_eligible": decision["passed"] and not dataset.get("synthetic", False),
        "limitations": [
            "Group and exact-content checks do not prove absence of near-duplicates",
            "Per-detection Wilson intervals assume independent trials; clustered scenes weaken this assumption",
            "Wall time includes preprocessing/loading, and model initialization for detection; not a sustained-stream deadline guarantee",
            "Dataset-level acceptance does not establish threat detection or performance on other domains",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.request, args.output), indent=2))
