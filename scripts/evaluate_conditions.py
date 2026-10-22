"""Bounded photometric robustness probes on existing labeled validation images.

These are transformed real-source images, not independent night/blur footage,
not a selection split, and not field-accuracy evidence.
"""

import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from surveilx.detector_expert import AdaptedYOLODetector, ScratchDetector
from training.datasets import digest
from training.detection_pipeline import evaluate_detections


def stratified_samples(samples, limit):
    """Cover available classes before filling; never imply unseen classes were probed."""
    if limit < 1:
        raise ValueError("Sample limit must be positive")
    shuffled = list(samples)
    random.Random(42).shuffle(shuffled)
    groups = {label: [] for sample in shuffled for label in sample["labels"]}
    for index, sample in enumerate(shuffled):
        for label in set(sample["labels"]):
            groups[label].append(index)
    selected = []
    empty = next((i for i, sample in enumerate(shuffled) if not sample["labels"]), None)
    if empty is not None:
        selected.append(empty)
    while len(selected) < min(limit, len(shuffled)):
        before = len(selected)
        for label in sorted(groups):
            index = next((i for i in groups[label] if i not in selected), None)
            if index is not None and len(selected) < limit:
                selected.append(index)
        if len(selected) == before:
            selected.extend(i for i in range(len(shuffled)) if i not in selected)
            break
    return [shuffled[i] for i in selected[:limit]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--samples", type=int, default=36)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = Path("data/runs") / args.version
    metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    dataset = next(
        p for p in Path("data/datasets").glob("*/manifest.json") if digest(p) == metadata["dataset_sha256"]
    )
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    samples = stratified_samples([s for s in manifest["samples"] if s["split"] == "validation"], args.samples)
    expert = (ScratchDetector if metadata["architecture"] == "sva-detector" else AdaptedYOLODetector)(run)
    rows = []
    for condition in ("original", "dark_gamma1.8", "mild_blur"):
        predictions = []
        truths = []
        for sample in samples:
            frame = cv2.imread(str(dataset.parent / sample["image"]))
            if condition == "dark_gamma1.8":
                frame = np.rint((frame.astype(np.float32) / 255) ** 1.8 * 255).astype(np.uint8)
            elif condition == "mild_blur":
                frame = cv2.GaussianBlur(frame, (3, 3), 1.0)
            found = expert.infer(frame, None)
            predictions.append(
                {
                    "boxes": torch.tensor([d.box for d in found]).reshape(-1, 4),
                    "labels": torch.tensor([metadata["classes"].index(d.label) for d in found]).long(),
                    "scores": torch.tensor([d.score for d in found]),
                }
            )
            truths.append(
                {
                    "boxes": torch.tensor(sample["boxes"]).reshape(-1, 4),
                    "labels": torch.tensor(sample["labels"]).long(),
                }
            )
        metrics = evaluate_detections(
            predictions, truths, len(metadata["classes"]), metadata["calibration"]["threshold"]
        )
        metrics.pop("map50")
        metrics.pop("map_iou")
        for row in metrics["per_class"]:
            row.pop("ap50")
            row.pop("ap_iou")
        metrics.pop("ap_definition")
        rows.append({"condition": condition, "metrics_at_frozen_calibration_threshold": metrics})
    output = Path("data/model-evaluations") / (args.version + "-conditions.json")
    output.write_text(
        json.dumps(
            {
                "version": args.version,
                "weights_sha256": metadata["weights_sha256"],
                "dataset_sha256": metadata["dataset_sha256"],
                "samples": len(samples),
                "sample_images": [s["image"] for s in samples],
                "sampling": "Deterministic class round robin, one empty image when available; seed42",
                "class_counts": {name: sum(s["labels"].count(index) for s in samples)
                    for index, name in enumerate(metadata["classes"])},
                "split": "validation_photometric_probe",
                "used_for_selection": False,
                "independent_acceptance": False,
                "scope": __doc__,
                "conditions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
