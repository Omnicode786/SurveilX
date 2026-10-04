"""Descriptive cluster-bootstrap uncertainty for held-out detection results."""

from collections import defaultdict

import numpy as np

from training.detection_pipeline import evaluate_detections


def interval(values, confidence=0.95):
    if not values:
        return {"lower": None, "upper": None, "confidence": confidence, "valid_resamples": 0,
                "status": "unavailable"}
    tail = (1 - confidence) / 2
    return {
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1 - tail)),
        "confidence": confidence,
        "valid_resamples": len(values),
        "status": "estimated" if len(values) >= 30 else "insufficient_resamples",
        "degenerate": bool(min(values) == max(values)),
    }


def cluster_bootstrap_detection(
    predictions,
    targets,
    groups,
    num_classes,
    score_threshold,
    iterations=500,
    seed=42,
):
    if not (len(predictions) == len(targets) == len(groups)):
        raise ValueError("Predictions, targets and groups must have the same length")
    if iterations < 1:
        raise ValueError("Positive resample count required")
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[str(group)].append(index)
    clusters = sorted(members)
    if len(clusters) < 8:
        return {
            "status": "insufficient_clusters",
            "clusters": len(clusters),
            "minimum_clusters": 8,
            "iterations": 0,
        }
    rng = np.random.default_rng(seed)
    supported = {int(label) for target in targets for label in target["labels"]}
    aggregate, per_class = [], [[] for _ in range(num_classes)]
    for _ in range(iterations):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        indexes = [index for group in sampled for index in members[group]]
        metrics = evaluate_detections(
            [predictions[index] for index in indexes],
            [targets[index] for index in indexes],
            num_classes,
            score_threshold,
        )
        present = {item["class_id"] for item in metrics["per_class"] if item["ground_truth"]}
        # Do not change the macro-AP taxonomy when a rare class vanishes in a resample.
        if supported and present == supported:
            aggregate.append(metrics["map50"])
        for item in metrics["per_class"]:
            if item["ap50"] is not None:
                per_class[item["class_id"]].append(item["ap50"])
    return {
        "status": "estimated",
        "clusters": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "map50": interval(aggregate),
        "per_class": [
            {"class_id": class_id, **interval(values)}
            for class_id, values in enumerate(per_class)
        ],
        "selection_criterion": False,
        "interpretation": "Percentile cluster bootstrap on untouched test data. Class intervals condition on that class being present; macro AP conditions on all originally supported classes being present. Inspect valid_resamples; fewer than 30 is insufficient. Recorded groups do not establish site independence; degenerate intervals do not imply perfect population performance.",
    }
