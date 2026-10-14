"""Read-only, complete-validation SVA failure attribution from frozen weights.

No optimizer, backward pass, calibration fit, checkpoint write or registration.
Raw-anchor localization is an upper bound, not detector recall or AP.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from training.detection_pipeline import (
    DetectionImages,
    calibrate_scores,
    calibrated_predictions,
    collate_detection,
    digest,
    evaluate_detections,
    resolve_asset,
)
from training.detector_model import (
    SVADetector,
    assign_targets,
    box_iou,
    decode_detections,
    detection_loss,
    flatten_predictions,
)


CATEGORIES = (
    "unlocalized",
    "localized_wrong_class",
    "correct_class_confidence_too_low",
    "duplicate_nms_or_matching",
    "correct_detection",
)


def matched_ground_truth(prediction, target, threshold, iou_threshold=0.5):
    """Use the production evaluator's score-ordered, class-aware one-to-one rule."""
    overlaps = box_iou(prediction["boxes"], target["boxes"])
    matched, duplicate_false_positives, other_false_positives = set(), 0, 0
    for position in prediction["scores"].argsort(descending=True).tolist():
        if float(prediction["scores"][position]) < threshold:
            continue
        compatible = [
            j for j, label in enumerate(target["labels"].tolist())
            if label == int(prediction["labels"][position]) and overlaps[position, j] >= iou_threshold
        ]
        available = [j for j in compatible if j not in matched]
        if available:
            matched.add(max(available, key=lambda j: float(overlaps[position, j])))
        elif compatible:
            duplicate_false_positives += 1
        else:
            other_false_positives += 1
    return matched, duplicate_false_positives, other_false_positives


def attribute_failures(raw, decoded, target, calibration, floor=0.001, iou_threshold=0.5):
    """Assign each GT exactly once; raw candidates retain their argmax class.

    A valid raw box at IoU >= .5 establishes the localization upper bound.
    Correct-class raw boxes must meet BOTH decoder floor and frozen calibrated
    threshold to count as eligible. Eligible raw boxes missing from the final
    one-to-one assignment are classified as decoder/matching losses, including
    pre-NMS top-k, NMS, final cap and overlapping-GT competition.
    """
    threshold = float(calibration["threshold"])
    calibrated = calibrated_predictions([decoded], calibration)[0]
    matched, duplicate_fp, other_fp = matched_ground_truth(calibrated, target, threshold, iou_threshold)
    boxes = raw["boxes"].clamp(0, 1)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    overlap = box_iou(target["boxes"], boxes)
    calibrated_raw = torch.from_numpy(calibrate_scores(raw["scores"].numpy(), calibration))
    rows = []
    for index, label in enumerate(target["labels"].tolist()):
        localized = (overlap[index] >= iou_threshold) & valid
        correct = localized & (raw["labels"] == label)
        eligible = correct & (raw["scores"] >= floor) & (calibrated_raw >= threshold)
        if index in matched:
            category = "correct_detection"
        elif not localized.any():
            category = "unlocalized"
        elif not correct.any():
            category = "localized_wrong_class"
        elif not eligible.any():
            category = "correct_class_confidence_too_low"
        else:
            category = "duplicate_nms_or_matching"
        best = int(overlap[index].masked_fill(~valid, -1).argmax()) if valid.any() else None
        rows.append({
            "class_id": label,
            "category": category,
            "best_iou": float(overlap[index, best]) if best is not None else 0.0,
            "best_iou_predicted_class": int(raw["labels"][best]) if best is not None else None,
            "best_iou_objectness": float(raw["objectness"][best]) if best is not None else None,
            "best_iou_true_class_probability": float(raw["class_probabilities"][best, label]) if best is not None else None,
            "best_correct_class_raw_score": float(raw["scores"][correct].max()) if correct.any() else None,
            "localized_raw_candidates": int(localized.sum()),
            "correct_class_raw_candidates": int(correct.sum()),
            "eligible_correct_class_raw_candidates": int(eligible.sum()),
        })
    return rows, {
        "raw_locations": len(raw["boxes"]),
        "raw_above_decoder_floor": int((valid & (raw["scores"] >= floor)).sum()),
        "decoded_before_operating_threshold": len(decoded["boxes"]),
        "decoded_above_operating_threshold": int((calibrated["scores"] >= threshold).sum()),
        "duplicate_false_positives": duplicate_fp,
        "other_false_positives": other_fp,
    }


def summarize(values):
    values = [value for value in values if value is not None]
    return {"count": len(values), "mean": float(np.mean(values)), "p10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))} if values else {"count": 0}


def diagnose(dataset, run, output):
    dataset, run, output = Path(dataset), Path(run), Path(output)
    manifest = json.loads(dataset.read_text())
    info = json.loads((run / "manifest.json").read_text())
    if info["config"].get("preprocessing", "stretch") != "stretch":
        raise ValueError("This historical g4 diagnostic requires stretch geometry; letterbox diagnostics need original-coordinate raw/decoder matching")
    if digest(dataset) != info["dataset_sha256"] or digest(run / "weights.pt") != info["weights_sha256"]:
        raise ValueError("Frozen dataset or weight integrity mismatch")
    if output.exists() or output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Diagnostic output must be new and outside the immutable model directory")
    assets = json.loads((run / "dataset_hashes.json").read_text())
    samples = [sample for sample in manifest["samples"] if sample["split"] == "validation"]
    for sample in samples:
        for key in ("image", "zone_map"):
            if sample.get(key) and digest(resolve_asset(dataset.parent, sample[key])) != assets[sample[key]]:
                raise ValueError(f"Frozen validation asset changed: {sample[key]}")
    torch.set_num_threads(3)
    model = SVADetector(**info["model_config"]).eval()
    model.load_state_dict(torch.load(run / "weights.pt", weights_only=True, map_location="cpu"))
    size = info["config"]["image_size"]
    images = DetectionImages(dataset.parent, samples, image_size=size, augment=False)
    counts, prediction_counts, confusion, levels = Counter(), Counter(), Counter(), Counter()
    by_class = {name: Counter({key: 0 for key in CATEGORIES}) for name in manifest["classes"]}
    per_class_rows, loss_rows, all_rows, predictions, targets = defaultdict(list), [], [], [], []
    gradient_sums = Counter()
    started = time.perf_counter()
    with torch.inference_mode():
        for sample, item in zip(samples, images, strict=True):
            batch, context, zones, target_batch = collate_detection([item])
            target = target_batch[0]
            heads = model(batch, context, zones)
            flat = flatten_predictions(heads)
            classes = flat["classes"][0].sigmoid()
            objectness = flat["objectness"][0, :, 0].sigmoid()
            quality_scoring = heads[0].get("quality_scores", False)
            scores, labels = (classes if quality_scoring else classes * objectness[:, None]).max(-1)
            raw = {"boxes": flat["boxes"][0], "scores": scores, "labels": labels,
                   "objectness": objectness, "class_probabilities": classes}
            decoded = decode_detections(heads, score_threshold=0.001)[0]
            rows, pred_counts = attribute_failures(raw, decoded, target, info["calibration"])
            prediction_counts.update(pred_counts)
            for row in rows:
                label = manifest["classes"][row["class_id"]]
                counts[row["category"]] += 1
                by_class[label][row["category"]] += 1
                by_class[label]["ground_truth"] += 1
                if row["best_iou"] >= 0.5:
                    predicted_label = manifest["classes"][row["best_iou_predicted_class"]]
                    confusion[(label, predicted_label)] += 1
                row["image"] = sample["image"]
                per_class_rows[label].append(row)
            all_rows.extend(rows)
            object_target, class_target, _ = assign_targets(flat, target_batch, assignment="coverage-v3" if quality_scoring else "center-v1")
            positive = object_target[0].bool()
            if positive.any() and not quality_scoring:
                probability = flat["classes"][0, positive].sigmoid()
                expected = class_target[0, positive]
                correct_probability = (probability * expected + (1 - probability) * (1 - expected)).clamp(1e-7, 1 - 1e-7)
                bce_gradient = 1 - correct_probability
                alpha = .25 * expected + .75 * (1 - expected)
                ratio = alpha * (1 - correct_probability) * ((1 - correct_probability) - 2 * correct_probability * correct_probability.log())
                for label, mask in (("true_class", expected.bool()), ("other_classes", ~expected.bool())):
                    gradient_sums[label + "_bce"] += float(bce_gradient[mask].sum())
                    gradient_sums[label + "_focal"] += float((bce_gradient * ratio)[mask].sum())
            levels.update({str(level): int((positive & (flat["levels"] == level)).sum()) for level in range(len(heads))})
            losses = {}
            for recipe, gamma in ((("dense_quality_v3", 0),) if quality_scoring else (("common_bce", 0), ("focal_gamma2_alpha025", 2))):
                measured = detection_loss(heads, target_batch,
                    classification_weight=info["config"].get("classification_weight", 1),
                    classification_focal_gamma=gamma, classification_focal_alpha=0.25)
                losses[recipe] = {key: float(measured[key]) for key in ("loss", "objectness", "classification", "box_giou", "box_l1")}
            loss_rows.append(losses)
            predictions.append({key: value for key, value in decoded.items() if key != "embeddings"})
            targets.append(target)
    calibration = info["calibration"]
    report = {
        "schema_version": 1, "scope": "Read-only full validation; eval/inference_mode CPU forward only; no fitting, backward, optimizer, training, activation or test-set selection",
        "dataset": str(dataset), "dataset_sha256": digest(dataset), "weights_sha256": info["weights_sha256"],
        "run": str(run), "images": len(samples), "image_size": size, "calibration": calibration,
        "definitions": {
            "categories": "Each GT belongs to exactly one of the five categories; see attribute_failures docstring",
            "localization": "Any geometrically valid raw anchor IoU >=0.50 regardless of confidence: oracle upper bound, not recall",
            "wrong_class": "No localized raw anchor has the correct argmax class",
            "confidence": "Correct-class raw localization exists but none passes raw floor .001 and frozen calibrated threshold",
            "decoder_matching": "Eligible correct-class raw localization lost through top600, classwise NMS .5, final100 or one-to-one competition",
            "confusion": "Class predicted by highest-IoU raw anchor, conditional on IoU >=.50; separate from exclusive counts",
            "losses": "Image-mean reference quality loss for v3, BCE/focal for legacy; not historical epoch losses or causal ablation",
            "score_mode": "dense_class_quality" if info.get("model_config", {}).get("architecture_version") == 3 else "objectness_times_class",
        },
        "counts": {key: counts[key] for key in CATEGORIES}, "ground_truth": len(all_rows),
        "per_class": {name: dict(values) for name, values in by_class.items()},
        "prediction_counts": dict(prediction_counts), "assignment_positives_per_level": dict(levels),
        "best_iou_confusion": [{"true": a, "predicted": b, "count": count} for (a, b), count in sorted(confusion.items())],
        "per_class_confidence": {name: {key: summarize([row[key] for row in rows]) for key in (
            "best_iou", "best_iou_objectness", "best_iou_true_class_probability", "best_correct_class_raw_score")}
            for name, rows in per_class_rows.items()},
        "loss_components": {recipe: {key: float(np.mean([row[recipe][key] for row in loss_rows]))
            for key in loss_rows[0][recipe]} for recipe in loss_rows[0]},
        "analytic_focal_to_bce_absolute_logit_gradient_ratio": {
            key: gradient_sums[key + "_focal"] / max(gradient_sums[key + "_bce"], 1e-12)
            for key in ("true_class", "other_classes")} if not quality_scoring else None,
        "validation_metrics": evaluate_detections(calibrated_predictions(predictions, calibration), targets,
            len(manifest["classes"]), calibration["threshold"]),
        "elapsed_seconds": time.perf_counter() - started,
        "ground_truth_details": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("run", "images", "ground_truth", "counts", "elapsed_seconds")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("run")
    parser.add_argument("output")
    diagnose(**vars(parser.parse_args()))
