"""Validation-only, fingerprinted comparison. Never promotes or activates models."""

import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.datasets import digest, validate_manifest
from training.detection_pipeline import (
    DetectionImages,
    collate_detection,
    predict,
    evaluate_detections,
    calibrated_predictions,
    detection_reliability,
)


def evaluate(run, dataset_path, output):
    metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest, _ = validate_manifest(dataset_path)
    if (
        digest(dataset_path) != metadata["dataset_sha256"]
        or digest(run / "weights.pt") != metadata["weights_sha256"]
    ):
        raise ValueError("Dataset or checkpoint checksum mismatch")
    if manifest["classes"] != metadata["classes"] or manifest["domain"] != metadata["domain"]:
        raise ValueError("Model/dataset contract mismatch")
    samples = [s for s in manifest["samples"] if s["split"] == "validation"]
    start = time.perf_counter()
    if metadata["task"] == "detection":
        from surveilx.detector_expert import ScratchDetector, AdaptedYOLODetector

        size = metadata["config"]["image_size"]
        if metadata["architecture"] == "sva-detector":
            expert = ScratchDetector(run)
            loader = DataLoader(
                DetectionImages(
                    dataset_path.parent,
                    samples,
                    size,
                    preprocessing=metadata["config"].get("preprocessing", "stretch"),
                ),
                batch_size=1,
                collate_fn=collate_detection,
            )
            predictions, targets, _ = predict(expert.model, loader, expert.device)
        else:
            from training.yolo_pipeline import predict_split

            expert = AdaptedYOLODetector(run)
            predictions, targets, _ = predict_split(
                expert.model,
                dataset_path,
                manifest,
                "validation",
                size,
                expert.device,
                metadata["config"].get("preprocessing", "stretch"),
            )
        predictions = calibrated_predictions(predictions, metadata["calibration"])
        metrics = evaluate_detections(
            predictions, targets, len(metadata["classes"]), metadata["calibration"]["threshold"]
        )
        # Reliability is computed on already calibrated scores exactly once.
        metrics["reliability"] = detection_reliability(predictions, targets)
        thresholds = [
            evaluate_detections(
                predictions,
                targets,
                len(metadata["classes"]),
                metadata["calibration"]["threshold"],
                round(float(iou), 2),
            )
            for iou in np.arange(0.5, 0.96, 0.05)
        ]
        metrics["map50_95"] = float(np.mean([r["map_iou"] for r in thresholds]))
        for i, row in enumerate(metrics["per_class"]):
            row["class_name"] = metadata["classes"][i]
            row["ap50_95"] = (
                float(np.mean([r["per_class"][i]["ap_iou"] for r in thresholds]))
                if row["ground_truth"]
                else None
            )
    else:
        from surveilx.expert import SVAExpert
        from training.pipeline import Clips, predict as event_predict
        from training.calibration import metrics as event_metrics

        expert = SVAExpert(run)
        loader = DataLoader(Clips(dataset_path.parent, samples, scene=expert.scene), batch_size=1)
        logits, labels = event_predict(expert.model, loader, expert.device)
        metrics = event_metrics(logits, labels, metadata["temperature"])
        for i, row in enumerate(metrics["per_class"]):
            row["class_name"] = metadata["classes"][i]
            row["missed_event_rate"] = 1 - row["recall"] if row["recall"] is not None else None
    elapsed = time.perf_counter() - start
    report = {
        "schema_version": 1,
        "version": run.name,
        "weights_sha256": metadata["weights_sha256"],
        "manifest_sha256": digest(run / "manifest.json"),
        "dataset_sha256": metadata["dataset_sha256"],
        "split": "validation",
        "classes": metadata["classes"],
        "task": metadata["task"],
        "domain": metadata["domain"],
        "metrics": metrics,
        "device": expert.device,
        "samples": len(samples),
        "end_to_end_ms_per_sample": elapsed * 1000 / len(samples),
        "latency_scope": "cold model load plus file I/O, preprocessing and forward; NOT steady-state runtime latency",
        "latency_usable_for_routing": False,
        "evaluated_at": time.time(),
        "independent_acceptance": False,
        "evaluator_sha256": digest(Path(__file__)),
        "metrics_code_sha256": digest(Path("training/detection_pipeline.py")),
    }
    output.mkdir(parents=True, exist_ok=True)
    temp = output / (run.name + ".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp.replace(output / (run.name + ".json"))
    del expert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--versions", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(2)
    datasets = {digest(p): p for p in (args.data / "datasets").glob("*/manifest.json")}
    output = args.data / "model-evaluations"
    seen = {}
    journal = []
    for path in sorted((args.data / "runs").glob("*/manifest.json")):
        if args.versions and path.parent.name not in args.versions:
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("task") not in ("detection", "event") or m.get("dataset_sha256") not in datasets:
            continue
        if not m.get("calibrated"):
            continue
        key = (
            m["weights_sha256"],
            m["dataset_sha256"],
            json.dumps(m.get("calibration", m.get("temperature")), sort_keys=True),
            json.dumps(m.get("config", {}).get("preprocessing", "stretch")),
            m.get("config", {}).get("image_size"),
        )
        try:
            cached_path = output / (path.parent.name + ".json")
            cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
            if (
                cached
                and cached.get("split") == "validation"
                and cached.get("weights_sha256") == m["weights_sha256"]
                and cached.get("dataset_sha256") == m["dataset_sha256"]
                and cached.get("manifest_sha256") == digest(path)
            ):
                assert digest(path.parent / "weights.pt") == m["weights_sha256"]
                report = cached
                seen[key] = cached
            elif key in seen:
                report = dict(
                    seen[key],
                    version=path.parent.name,
                    manifest_sha256=digest(path),
                    reused_identical_evaluation=True,
                )
                assert digest(path.parent / "weights.pt") == m["weights_sha256"]
                (output / (path.parent.name + ".json")).write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
            else:
                report = evaluate(path.parent, datasets[m["dataset_sha256"]], output)
                seen[key] = report
            journal.append({"version": path.parent.name, "state": "completed"})
            print(
                json.dumps(
                    {
                        "version": path.parent.name,
                        "state": "completed",
                        "validation": report["metrics"].get("map50", report["metrics"].get("macro_f1")),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            journal.append({"version": path.parent.name, "state": "failed", "error": str(exc)})
            print(json.dumps(journal[-1]), flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "status.json").write_text(json.dumps(journal, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
