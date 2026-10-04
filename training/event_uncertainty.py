"""Held-out event uncertainty; recorded groups are not proof of site independence."""

from statistics import NormalDist

import numpy as np


def wilson(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("Valid success/trial counts and confidence are required")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, scale = successes / trials, 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / scale
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / scale
    return {
        "lower": max(0.0, float(center - half)),
        "upper": min(1.0, float(center + half)),
        "confidence": confidence,
    }


def event_uncertainty(logits, labels, groups, iterations=500, seed=42):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if (
        logits.ndim != 2
        or logits.shape[1] < 2
        or not np.isfinite(logits).all()
        or labels.ndim != 1
        or len(logits) != len(labels)
        or len(groups) != len(labels)
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= logits.shape[1])
        or any(not isinstance(group, str) or not group for group in groups)
        or iterations < 100
    ):
        raise ValueError("Finite logits, integer labels, recorded groups and >=100 resamples required")
    unique = sorted(set(groups))
    members = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    correct = logits.argmax(1) == labels
    all_correct = sum(bool(correct[indexes].all()) for indexes in members)
    report = {
        "samples": len(labels),
        "recorded_groups": len(unique),
        "selection_criterion": False,
        "group_all_correct": {
            "successes": all_correct,
            "trials": len(unique),
            "interval": wilson(all_correct, len(unique)),
            "estimand": "Probability every clip in a recorded group is correct; not clip accuracy",
        },
        "limitations": "Intervals assume representative independent recorded groups. Shared actors/sites violate that assumption. Perfect empirical bootstrap scores do not imply perfect population accuracy.",
    }
    if len(unique) < 8:
        return {**report, "status": "insufficient_clusters", "minimum_clusters": 8, "iterations": 0}
    rng = np.random.default_rng(seed)
    aggregate, recalls = [], [[] for _ in range(logits.shape[1])]
    for _ in range(iterations):
        indexes = np.concatenate([members[index] for index in rng.integers(0, len(members), len(members))])
        aggregate.append(correct[indexes].mean())
        for class_id, values in enumerate(recalls):
            selected = indexes[labels[indexes] == class_id]
            if len(selected):
                values.append(float(correct[selected].mean()))

    def percentile(values):
        return {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence": 0.95,
            "valid_resamples": len(values),
            "degenerate": bool(min(values) == max(values)),
        }

    return {
        **report,
        "status": "estimated",
        "iterations": iterations,
        "seed": seed,
        "accuracy_cluster_bootstrap": percentile(aggregate),
        "per_class_recall": [
            {
                "class_id": index,
                "support": int((labels == index).sum()),
                "interval": percentile(values) if values else None,
            }
            for index, values in enumerate(recalls)
        ],
    }
