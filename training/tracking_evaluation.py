"""Read-only multi-class tracking evaluation on explicitly labeled sequences.

CLEAR-style temporal matching and global identity assignment. Development
metrics, not an official MOTChallenge/TrackEval dataset adapter or HOTA.
References and format: docs/tracking-evaluation.md.
"""
from collections import Counter
import math

import numpy as np

from surveilx.matching import maximum_assignment
from surveilx.vision import iou


FIELDS = ("gt", "predicted", "matched", "false_positives", "false_negatives", "id_switches", "id_true_positives")


def validate_objects(objects, identities):
    seen = set()
    for obj in objects:
        key, label, box = obj.get("track_id"), obj.get("label"), obj.get("box")
        if isinstance(key, bool) or not isinstance(key, (str, int)) or key == "" or not isinstance(label, str) or not label:
            raise ValueError("Each object requires a track_id and nonempty label")
        if key in seen:
            raise ValueError("Duplicate identity within frame")
        seen.add(key)
        if key in identities and identities[key] != label:
            raise ValueError("An identity cannot change class within a sequence")
        identities[key] = label
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in box):
            raise ValueError("Tracking boxes must be finite normalized xyxy")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Tracking boxes must have positive area")


def scores(counts):
    gt, pred, matched = counts["gt"], counts["predicted"], counts["matched"]
    idtp = counts["id_true_positives"]
    return {**{key: counts[key] for key in FIELDS},
            "mota": 1 - (counts["false_negatives"] + counts["false_positives"] + counts["id_switches"]) / gt if gt else None,
            "motp_mean_iou": counts["matched_iou_sum"] / matched if matched else None,
            "precision": matched / pred if pred else None, "recall": matched / gt if gt else None,
            "id_false_positives": pred - idtp, "id_false_negatives": gt - idtp,
            "idf1": 2 * idtp / (gt + pred) if gt + pred else None}


def evaluate_tracking(payload, threshold=.5):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1 or not isinstance(payload.get("sequences"), list) or not payload["sequences"]:
        raise ValueError("Expected schema_version 1 and nonempty sequences")
    if any(key in payload and not isinstance(payload[key], bool) for key in ("synthetic", "ground_truth_reviewed")):
        raise ValueError("Evidence flags must be booleans")
    total, classes, results, names = Counter(), {}, [], set()
    for sequence in payload["sequences"]:
        if not isinstance(sequence, dict):
            raise ValueError("Sequences must be objects")
        name, frames = sequence.get("name"), sequence.get("frames")
        if not isinstance(name, str) or not name or name in names or not isinstance(frames, list) or not frames:
            raise ValueError("Sequences require unique names and nonempty frame lists")
        names.add(name)
        previous, last_match, gt_labels, pred_labels = {}, {}, {}, {}
        gt_counts, pred_counts, potential, per_class = Counter(), Counter(), Counter(), {}
        last_index = None
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Frames must be objects")
            index = frame.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or (last_index is not None and index != last_index + 1):
                raise ValueError("Frame indices must be consecutive integers; include empty frames explicitly")
            last_index = index
            gt, pred = frame.get("ground_truth"), frame.get("predictions")
            if not isinstance(gt, list) or not isinstance(pred, list):
                raise ValueError("Each frame requires ground_truth and predictions lists")
            validate_objects(gt, gt_labels)
            validate_objects(pred, pred_labels)
            for obj in gt + pred:
                per_class.setdefault(obj["label"], Counter())
            overlap, weights = np.zeros((len(gt), len(pred))), np.zeros((len(gt), len(pred)))
            for row, truth in enumerate(gt):
                key = truth["track_id"]
                gt_counts[key] += 1
                per_class[truth["label"]]["gt"] += 1
                for column, detection in enumerate(pred):
                    if truth["label"] == detection["label"]:
                        similarity = iou(truth["box"], detection["box"])
                        overlap[row, column] = similarity
                        if similarity >= threshold:
                            potential[key, detection["track_id"]] += 1
                            weights[row, column] = 1000 * (previous.get(key) == detection["track_id"]) + similarity
            for detection in pred:
                pred_counts[detection["track_id"]] += 1
                per_class[detection["label"]]["predicted"] += 1
            current = {}
            for row, column in maximum_assignment(weights):
                if weights[row, column] == 0:
                    continue
                truth, detection = gt[row], pred[column]
                key, identity = truth["track_id"], detection["track_id"]
                c = per_class[truth["label"]]
                c["matched"] += 1
                c["matched_iou_sum"] += overlap[row, column]
                c["id_switches"] += int(key in last_match and last_match[key] != identity)
                current[key] = last_match[key] = identity
            previous = current  # Empty/unmatched frames break continuity preference.
        gt_ids, pred_ids = list(gt_counts), list(pred_counts)
        weights = np.array([[potential[g, p] for p in pred_ids] for g in gt_ids]).reshape(len(gt_ids), len(pred_ids))
        for row, column in maximum_assignment(weights):
            per_class[gt_labels[gt_ids[row]]]["id_true_positives"] += int(weights[row, column])
        accumulated = Counter()
        for label, c in per_class.items():
            c["false_positives"], c["false_negatives"] = c["predicted"] - c["matched"], c["gt"] - c["matched"]
            accumulated.update(c)
            classes.setdefault(label, Counter()).update(c)
        total.update(accumulated)
        results.append({"name": name, "frames": len(frames), "metrics": scores(accumulated),
                        "per_class": {label: scores(c) for label, c in sorted(per_class.items())}})
    return {"schema_version": 1, "scope": "Offline labeled-sequence development evaluation; no model training or activation",
            "matching": "class-aware IoU; previous-frame continuity preference; switches against last matched identity across gaps",
            "iou_threshold": threshold, "synthetic": bool(payload.get("synthetic", False)),
            "ground_truth_reviewed": payload.get("ground_truth_reviewed") is True,
            "metrics": scores(total), "per_class": {label: scores(c) for label, c in sorted(classes.items())}, "sequences": results}
