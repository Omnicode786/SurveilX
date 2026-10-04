"""Inspect scratch localization versus classification on training/validation samples."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.detection_pipeline import DetectionImages, collate_detection, digest
from training.detector_model import SVADetector, box_iou, decode_detections, detection_loss


def diagnostic_samples(samples, classes, limit):
    """Deterministic class round-robin avoids filename-ordered single-class probes."""
    buckets = [
        [index for index, sample in enumerate(samples) if label in sample["labels"]]
        for label in range(classes)
    ]
    buckets.append([index for index, sample in enumerate(samples) if not sample["labels"]])
    selected, seen = [], set()
    for offset in range(max(map(len, buckets), default=0)):
        for bucket in buckets:
            if offset < len(bucket) and bucket[offset] not in seen:
                seen.add(bucket[offset])
                selected.append(samples[bucket[offset]])
                if len(selected) == limit:
                    return selected
    return selected


def diagnose(dataset, run, output, limit=64):
    if not 1 <= limit <= 256:
        raise ValueError("Diagnostic limit must be between 1 and 256")
    dataset, run = Path(dataset), Path(run)
    manifest = json.loads(dataset.read_text())
    model_info = json.loads((run / "manifest.json").read_text())
    if digest(run / "weights.pt") != model_info["weights_sha256"]:
        raise ValueError("Model integrity mismatch")
    if digest(dataset) != model_info["dataset_sha256"]:
        raise ValueError("Diagnostic dataset must match the frozen model")
    torch.set_num_threads(1)
    model = SVADetector(**model_info["model_config"]).eval()
    model.load_state_dict(torch.load(run / "weights.pt", weights_only=True, map_location="cpu"))
    size = model_info["config"]["image_size"]
    result = {
        "dataset": str(dataset),
        "model": str(run),
        "image_size": size,
        "scope": "Class-stratified train/validation localization diagnostic; not representative AP or independent acceptance",
        "weights_sha256": model_info["weights_sha256"],
        "splits": {},
    }
    for split in ("train", "validation"):
        samples = diagnostic_samples(
            [row for row in manifest["samples"] if row["split"] == split], len(manifest["classes"]), limit
        )
        loader = DataLoader(
            DetectionImages(dataset.parent, samples, image_size=size),
            batch_size=1,
            collate_fn=collate_detection,
        )
        counts, predicted, losses = Counter(), Counter(), []
        classes = {name: Counter() for name in manifest["classes"]}
        with torch.no_grad():
            for images, context, zones, targets in loader:
                heads = model(images, context, zones)
                loss = detection_loss(
                    heads,
                    targets,
                    classification_weight=model_info["config"].get("classification_weight", 1),
                    classification_focal_gamma=model_info["config"].get("classification_focal_gamma", 0),
                    classification_focal_alpha=model_info["config"].get("classification_focal_alpha", 0.25),
                )
                losses.append({key: float(value) for key, value in loss.items()})
                for prediction, target in zip(
                    decode_detections(heads, score_threshold=0.001), targets, strict=True
                ):
                    predicted.update(int(label) for label in prediction["labels"])
                    counts["ground_truth"] += len(target["labels"])
                    for label in target["labels"].tolist():
                        classes[manifest["classes"][label]]["ground_truth"] += 1
                    if not len(prediction["boxes"]) or not len(target["boxes"]):
                        continue
                    overlap = box_iou(target["boxes"], prediction["boxes"])
                    values, locations = overlap.max(1)
                    localized = values >= 0.5
                    counts["localized_gt_iou50"] += int(localized.sum())
                    counts["correct_class_localized_gt"] += int(
                        (localized & (prediction["labels"][locations] == target["labels"])).sum()
                    )
                    for index, label in enumerate(target["labels"].tolist()):
                        row = classes[manifest["classes"][label]]
                        row["localized_gt_iou50"] += int(localized[index])
                        row["correct_class_localized_gt"] += int(
                            localized[index] and prediction["labels"][locations[index]] == label
                        )
        result["splits"][split] = {
            "images": len(samples),
            "counts": dict(counts),
            "per_class": {name: dict(values) for name, values in classes.items()},
            "prediction_labels": {manifest["classes"][key]: value for key, value in predicted.items()},
            "loss_components": {key: sum(row[key] for row in losses) / len(losses) for key in losses[0]},
        }
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("run")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int, default=64)
    diagnose(**vars(parser.parse_args()))
