"""Reproducible image detection training, evaluation, and held-out calibration."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from surveilx.accelerators import torch_device
from training.detector_model import SVADetector, box_iou, decode_detections, detection_loss

SPLITS = ("train", "validation", "calibration", "test")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolve_asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Dataset asset is missing or outside the dataset root: {relative}")
    return path


def validate_detection_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    from surveilx.task_profiles import validate_profiles

    validate_profiles(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("task") != "detection":
        raise ValueError("Expected a schema_version=1 detection manifest")
    if not manifest.get("domain") or not manifest.get("license"):
        raise ValueError("Detection datasets require a domain and license/rights record")
    classes = manifest.get("classes", [])
    if not classes or len(classes) != len(set(classes)) or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("classes must contain distinct nonempty class names")
    if not manifest.get("samples"):
        raise ValueError("The dataset has no labeled samples")
    counts, groups, images, hashes = dict.fromkeys(SPLITS, 0), {}, {}, {}
    for sample in manifest["samples"]:
        split, group = sample.get("split"), sample.get("group")
        if split not in SPLITS or not isinstance(group, str) or not group:
            raise ValueError("Every sample needs an explicit split and source group")
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group leaks across dataset splits: {group}")
        groups[group] = split
        image = resolve_asset(path.parent, sample["image"])
        image_hash = digest(image)
        if image_hash in images and images[image_hash] != split:
            raise ValueError("Identical image content leaks across dataset splits")
        if sample.get("sha256") and sample["sha256"] != image_hash:
            raise ValueError(f"Image checksum mismatch: {sample['image']}")
        images[image_hash], hashes[sample["image"]] = split, image_hash
        boxes = np.asarray(sample.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(sample.get("labels", []))
        if len(boxes) != len(labels) or not np.isfinite(boxes).all():
            raise ValueError("Each box needs a class label and finite coordinates")
        if len(boxes) and (np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Bounding boxes must be normalized xyxy with positive area")
        if len(labels) and (
            not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= len(classes))
        ):
            raise ValueError("Class labels must be zero-based integers from classes")
        context = np.asarray(sample.get("context", [0, 0, 0, 0]), dtype=np.float32)
        if context.shape != (4,) or not np.isfinite(context).all():
            raise ValueError("context must have exactly four finite numbers")
        if sample.get("zone_map"):
            zone = resolve_asset(path.parent, sample["zone_map"])
            hashes[sample["zone_map"]] = digest(zone)
        counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Training requires nonempty train, validation, calibration, and test splits")
    return manifest, counts, hashes


class DetectionImages(Dataset):
    def __init__(self, root, samples, image_size=256, augment=False):
        self.root, self.samples, self.image_size, self.augment = Path(root), samples, image_size, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(resolve_asset(self.root, sample["image"])))
        if image is None:
            raise ValueError(f"Cannot decode image: {sample['image']}")
        image = cv2.cvtColor(cv2.resize(image, (self.image_size, self.image_size)), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255
        boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 4).copy()
        zone = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        if sample.get("zone_map"):
            source = cv2.imread(str(resolve_asset(self.root, sample["zone_map"])), cv2.IMREAD_GRAYSCALE)
            if source is None:
                raise ValueError(f"Cannot decode zone map: {sample['zone_map']}")
            zone = cv2.resize(source, (self.image_size, self.image_size)).astype(np.float32) / 255
        if self.augment:
            # Explicit context may encode left/right orientation; do not silently invalidate it.
            if not sample.get("context") and random.random() < 0.5:
                image, zone = image[:, ::-1].copy(), zone[:, ::-1].copy()
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
            image = np.clip(image * random.uniform(0.8, 1.2) + random.uniform(-0.06, 0.06), 0, 1)
        return (
            torch.from_numpy(image.transpose(2, 0, 1).copy()),
            torch.tensor(sample.get("context", [0, 0, 0, 0]), dtype=torch.float32),
            torch.from_numpy(zone[None].copy()),
            {"boxes": torch.from_numpy(boxes), "labels": torch.tensor(sample["labels"], dtype=torch.long)},
        )


def collate_detection(batch):
    images, context, zones, targets = zip(*batch)
    return torch.stack(images), torch.stack(context), torch.stack(zones), list(targets)


def match_predictions(predictions, targets, iou_threshold=0.5):
    """Score-ordered, class-aware one-to-one matching independently in each image."""
    records = []
    for image_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        boxes = torch.as_tensor(prediction["boxes"], dtype=torch.float32).cpu().reshape(-1, 4)
        scores = torch.as_tensor(prediction["scores"], dtype=torch.float32).cpu()
        labels = torch.as_tensor(prediction["labels"], dtype=torch.long).cpu()
        true_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).cpu().reshape(-1, 4)
        true_labels = torch.as_tensor(target["labels"], dtype=torch.long).cpu()
        used = set()
        overlaps = box_iou(boxes, true_boxes)
        for position in scores.argsort(descending=True).tolist():
            candidates = [
                j
                for j, label in enumerate(true_labels.tolist())
                if label == int(labels[position]) and j not in used
            ]
            correct = False
            if candidates:
                best = max(candidates, key=lambda j: float(overlaps[position, j]))
                if float(overlaps[position, best]) >= iou_threshold:
                    used.add(best)
                    correct = True
            records.append(
                {
                    "score": float(scores[position]),
                    "label": int(labels[position]),
                    "correct": correct,
                    "image": image_index,
                }
            )
    return records


def evaluate_detections(predictions, targets, num_classes, score_threshold=0.25):
    """Shared evaluator: tensor/array lists of boxes/scores/labels, normalized xyxy.

    AP50 uses 101-point interpolated precision, per-class matching at IoU >= .5.
    Prediction input should use a low score floor (e.g. .001), not display filtering.
    """
    records = match_predictions(predictions, targets)
    counts = np.zeros(num_classes, dtype=int)
    for target in targets:
        for label in torch.as_tensor(target["labels"]).tolist():
            counts[label] += 1
    per_class = []
    for label in range(num_classes):
        selected = sorted((r for r in records if r["label"] == label), key=lambda r: -r["score"])
        correct = np.array([r["correct"] for r in selected], dtype=np.float64)
        cumulative = np.cumsum(correct)
        precision = cumulative / np.maximum(np.arange(len(correct)) + 1, 1)
        recall = cumulative / max(int(counts[label]), 1)
        ap = float(np.mean([np.max(precision[recall >= q], initial=0) for q in np.linspace(0, 1, 101)]))
        per_class.append(
            {"class_id": label, "ground_truth": int(counts[label]), "ap50": ap if counts[label] else None}
        )
    accepted = [r for r in records if r["score"] >= score_threshold]
    tp = sum(r["correct"] for r in accepted)
    precision, recall = tp / max(len(accepted), 1), tp / max(int(counts.sum()), 1)
    aps = [item["ap50"] for item in per_class if item["ap50"] is not None]
    return {
        "map50": float(np.mean(aps)) if aps else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "true_positives": tp,
        "false_positives": len(accepted) - tp,
        "false_negatives": int(counts.sum()) - tp,
        "ground_truth": int(counts.sum()),
        "score_threshold": score_threshold,
        "iou_threshold": 0.5,
        "per_class": per_class,
        "ap_definition": "101-point interpolated precision at IoU 0.50, class-aware one-to-one matching",
    }


def calibrate_scores(scores, calibration):
    scores = np.asarray(scores, dtype=np.float64).clip(1e-7, 1 - 1e-7)
    logits = np.log(scores / (1 - scores))
    adjusted = logits / calibration.get("temperature", 1) + calibration.get("bias", 0)
    return 1 / (1 + np.exp(-np.clip(adjusted, -50, 50)))


def detection_reliability(predictions, targets, calibration=None):
    records = match_predictions(predictions, targets)
    if not records:
        return {"samples": 0, "ece": None, "brier": None, "nll": None}
    scores = np.array([r["score"] for r in records], dtype=float)
    labels = np.array([r["correct"] for r in records], dtype=float)
    if calibration:
        scores = calibrate_scores(scores, calibration)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    bins = np.minimum((scores * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == i) * abs(scores[bins == i].mean() - labels[bins == i].mean())
        for i in range(10)
        if np.any(bins == i)
    )
    return {
        "samples": len(records),
        "ece": float(ece),
        "brier": float(np.mean((scores - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))),
        "scope": "Retained predictions at score floor .001; does not penalize missed objects",
    }


def fit_detection_calibration(predictions, targets):
    records = match_predictions(predictions, targets)
    scores = np.array([r["score"] for r in records])
    correct = np.array([r["correct"] for r in records], dtype=float)
    result = {
        "temperature": 1.0,
        "bias": 0.0,
        "threshold": 0.25,
        "status": "insufficient_correct_and_incorrect_predictions",
        "samples": len(records),
        "task": "detection_correctness_at_iou_0.50",
        "split": "calibration",
    }
    if len(records) < 20 or correct.sum() < 3 or (1 - correct).sum() < 3:
        return result
    logits = np.log(scores.clip(1e-7, 1 - 1e-7) / (1 - scores.clip(1e-7, 1 - 1e-7)))
    # Bounded grid fit is deterministic and monotone, preserving detection/AP rank order.
    best = float("inf")
    for temperature in np.geomspace(0.25, 8, 32):
        for bias in np.linspace(-6, 6, 49):
            z = logits / temperature + bias
            nll = float(np.mean(np.logaddexp(0, z) - correct * z))
            if nll < best:
                best = nll
                result.update(temperature=float(temperature), bias=float(bias))
    probabilities = calibrate_scores(scores, result)
    total_truth = sum(len(t["labels"]) for t in targets)
    best_f1 = -1
    for threshold in np.unique(probabilities):
        selected = probabilities >= threshold
        tp = correct[selected].sum()
        f1 = 2 * tp / max(selected.sum() + total_truth, 1)
        if f1 > best_f1:
            best_f1 = f1
            result["threshold"] = float(threshold)
    result.update(status="fitted", nll=best, calibration_f1=float(best_f1))
    return result


def calibrated_predictions(predictions, calibration):
    return [
        {
            **p,
            "scores": torch.tensor(
                calibrate_scores(p["scores"].cpu().numpy(), calibration), dtype=torch.float32
            ),
        }
        for p in predictions
    ]


@torch.no_grad()
def predict(model, loader, device, measure_loss=False):
    model.eval()
    predictions, targets, losses, latencies = [], [], [], []
    for images, context, zones, target in loader:
        images, context, zones = images.to(device), context.to(device), zones.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(images, context, zones)
        decoded = decode_detections(output, score_threshold=0.001)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000 / len(images))
        predictions.extend([{k: v.cpu() for k, v in item.items() if k != "embeddings"} for item in decoded])
        targets.extend(target)
        if measure_loss:
            losses.append(float(detection_loss(output, target)["loss"]))
    return (
        predictions,
        targets,
        {
            "loss": float(np.mean(losses)) if losses else None,
            "mean_batch_amortized_latency_ms": float(np.mean(latencies)),
            "latency_scope": "forward+decode+NMS; excludes file loading and host-to-device copy",
        },
    )


def train(
    manifest_path,
    output,
    epochs=10,
    seed=42,
    image_size=256,
    width=24,
    depth=2,
    batch_size=8,
    learning_rate=0.001,
    threads=4,
    initialize_from=None,
):
    if epochs < 1 or batch_size < 1 or image_size < 64 or image_size % 32:
        raise ValueError("epochs/batch_size must be positive; image_size must be a multiple of 32 >=64")
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, counts, asset_hashes = validate_detection_manifest(manifest_path)
    if output.exists() and any(p.name != "process.log" for p in output.iterdir()):
        raise ValueError("Output run already contains files; choose a new version")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(threads, torch.get_num_threads())))
    device = torch_device()
    model = SVADetector(classes=len(manifest["classes"]), width=width, depth=depth).to(device)
    initialization = "random"
    if initialize_from:
        previous = Path(initialize_from).resolve()
        previous_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        if (
            previous_manifest.get("classes") != manifest["classes"]
            or previous_manifest.get("model_config") != model.config
        ):
            raise ValueError("Continuation requires identical class order and architecture")
        if digest(previous / "weights.pt") != previous_manifest["weights_sha256"]:
            raise ValueError("Continuation checkpoint checksum mismatch")
        model.load_state_dict(torch.load(previous / "weights.pt", map_location=device, weights_only=True))
        initialization = previous_manifest["weights_sha256"]
    loaders = {
        split: DataLoader(
            DetectionImages(
                manifest_path.parent,
                [s for s in manifest["samples"] if s["split"] == split],
                image_size=image_size,
                augment=split == "train",
            ),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            collate_fn=collate_detection,
        )
        for split in SPLITS
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )
    started, history, best, best_epoch = time.perf_counter(), [], (-1.0, -float("inf")), 0
    run_config = {
        "epochs": epochs,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "threads": threads,
        "initialization": initialization,
    }
    (output / "config.json").write_text(
        json.dumps({**run_config, "model": model.config}, indent=2), encoding="utf-8"
    )
    (output / "dataset_hashes.json").write_text(json.dumps(asset_hashes, indent=2), encoding="utf-8")
    for epoch in range(epochs):
        model.train()
        losses = []
        for images, context, zones, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            result = detection_loss(model(images.to(device), context.to(device), zones.to(device)), targets)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("Non-finite training loss; candidate not saved as successful")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(result["loss"].detach()))
        predictions, targets, validation = predict(model, loaders["validation"], device, measure_loss=True)
        metrics = evaluate_detections(predictions, targets, len(manifest["classes"]))
        score = (metrics["map50"] or 0, -validation["loss"])
        if score > best:
            best, best_epoch = score, epoch + 1
            torch.save(model.state_dict(), output / "weights.pt")
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "validation_map50": metrics["map50"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        scheduler.step()
    model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
    predictions, targets, _ = predict(model, loaders["calibration"], device)
    calibration = fit_detection_calibration(predictions, targets)
    predictions, targets, timing = predict(model, loaders["test"], device)
    raw_metrics = evaluate_detections(predictions, targets, len(manifest["classes"]))
    calibrated = calibrated_predictions(predictions, calibration)
    metrics = evaluate_detections(calibrated, targets, len(manifest["classes"]), calibration["threshold"])
    serial_predictions = [{k: v.tolist() for k, v in p.items()} for p in predictions]
    (output / "test_predictions.json").write_text(json.dumps(serial_predictions, indent=2), encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "task": "detection",
        "architecture": "sva-detector",
        "name": "SVA-Detector",
        "stage": "candidate",
        "model_config": model.config,
        "config": run_config,
        "classes": manifest["classes"],
        "domain": manifest.get("domain", "unspecified"),
        "capability_ids": manifest.get("capability_ids", []),
        "synthetic": manifest.get("synthetic", False),
        "license": manifest.get("license"),
        "dataset_sha256": digest(manifest_path),
        "dataset_content_sha256": digest(output / "dataset_hashes.json"),
        "weights_sha256": digest(output / "weights.pt"),
        "dataset_counts": counts,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "calibration": calibration,
        "calibrated": calibration["status"] == "fitted",
        "metrics": metrics,
        "uncalibrated_metrics": raw_metrics,
        "raw_test_reliability": detection_reliability(predictions, targets),
        "calibrated_test_reliability": detection_reliability(predictions, targets, calibration),
        "timing": timing,
        "history": history,
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "duration_seconds": time.perf_counter() - started,
        "deployment_eligible": False,
        "gate_reason": "Experimental candidate; requires independent domain, latency, calibration and class coverage acceptance",
        "trained_heads": ["boxes", "objectness", "classes"],
        "context_supervision": "provided"
        if any(s.get("context") or s.get("zone_map") for s in manifest["samples"])
        else "absent; zero context",
        "outperforms_yolo": "not established; requires matched held-out benchmark and hardware budget",
    }
    (output / "manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("manifest")
    training.add_argument("output")
    for flag, default in (
        ("epochs", 10),
        ("seed", 42),
        ("image-size", 256),
        ("width", 24),
        ("depth", 2),
        ("batch-size", 8),
        ("threads", 4),
    ):
        training.add_argument(f"--{flag}", type=int, default=default)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--initialize-from")
    args = vars(parser.parse_args())
    args.pop("command")
    args["manifest_path"] = args.pop("manifest")
    print(json.dumps(train(**args), indent=2))


if __name__ == "__main__":
    main()
