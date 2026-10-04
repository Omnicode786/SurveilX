"""Statistical input/output drift signals that request review, never create labels."""

import math
import time
from collections import defaultdict

import numpy as np
from sqlalchemy import select

from surveilx.database import Record, transaction


FEATURES = {
    "motion": (np.array([-np.inf, 0.01, 0.05, 0.15, 0.35, 0.65, np.inf]), lambda p: p["state"]["motion"]),
    "brightness": (
        np.array([-np.inf, 0.1, 0.25, 0.45, 0.65, 0.85, np.inf]),
        lambda p: p["state"]["brightness"],
    ),
    "latency_ms": (
        np.array([-np.inf, 25, 50, 100, 200, 400, 800, np.inf]),
        lambda p: p["latency_ms"],
    ),
    "detection_count": (
        np.array([-np.inf, 0.5, 1.5, 3.5, 7.5, 15.5, np.inf]),
        lambda p: len(p.get("result", [])),
    ),
    "mean_confidence": (
        np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]),
        lambda p: np.mean([row.get("confidence", 0) for row in p.get("result", [])])
        if p.get("result")
        else 0,
    ),
}


def distribution(values, bins):
    counts = np.histogram(values, bins=bins)[0].astype(float)
    # Jeffreys smoothing makes the divergence finite without erasing empty-bin changes.
    counts += 0.5
    return counts / counts.sum()


def jensen_shannon(first, second, bins):
    """Base-2 Jensen-Shannon divergence, bounded to [0, 1]."""
    p, q = distribution(first, bins), distribution(second, bins)
    middle = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / middle)) + 0.5 * np.sum(q * np.log2(q / middle)))


def permutation_pvalue(first, second, bins, observed, permutations=199, seed=42):
    """Two-sample randomization test for a histogram-divergence effect."""
    combined = np.concatenate([first, second])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(combined)
        effect = jensen_shannon(shuffled[: len(first)], shuffled[len(first) :], bins)
        exceed += effect + 1e-12 >= observed
    return float((exceed + 1) / (permutations + 1))


def identity(payload):
    return (
        payload.get("camera_id", "unknown"),
        payload.get("expert_slot", "baseline"),
        payload.get("model_version") or payload.get("action", "unknown"),
    )


def feature_values(rows, feature):
    accessor = FEATURES[feature][1]
    values = []
    for row in rows:
        try:
            value = float(accessor(row))
            if math.isfinite(value):
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(values, dtype=float)


def analyze_group(rows, reference_size=60, recent_size=30, effect_floor=0.10, alpha=0.05):
    """Compare fixed, non-overlapping windows and correct for testing five features."""
    required = reference_size + recent_size
    if len(rows) < required:
        return {"state": "collecting", "observations": len(rows), "required": required, "features": []}
    reference, recent = rows[-required:-recent_size], rows[-recent_size:]
    results = []
    corrected_alpha = alpha / len(FEATURES)
    for index, (name, (bins, _)) in enumerate(FEATURES.items()):
        before, after = feature_values(reference, name), feature_values(recent, name)
        if len(before) != reference_size or len(after) != recent_size:
            continue
        effect = jensen_shannon(before, after, bins)
        pvalue = permutation_pvalue(before, after, bins, effect, seed=42 + index)
        results.append(
            {
                "feature": name,
                "js_divergence": effect,
                "p_value": pvalue,
                "effect_floor": effect_floor,
                "corrected_alpha": corrected_alpha,
                "shift": effect >= effect_floor and pvalue <= corrected_alpha,
            }
        )
    shifted = [row for row in results if row["shift"]]
    complete = len(results) == len(FEATURES)
    return {
        "state": "review_required" if shifted else "stable" if complete else "collecting",
        "observations": required,
        "reference_size": reference_size,
        "recent_size": recent_size,
        "features": results,
        "shifted_features": [row["feature"] for row in shifted],
        "reference_period": [reference[0].get("timestamp"), reference[-1].get("timestamp")],
        "recent_period": [recent[0].get("timestamp"), recent[-1].get("timestamp")],
    }


def analyze_decisions(payloads, **options):
    groups = defaultdict(list)
    for payload in payloads:
        groups[identity(payload)].append(payload)
    streams = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row.get("timestamp", 0))
        streams.append(
            {
                "camera_id": key[0],
                "expert_slot": key[1],
                "model": key[2],
                **analyze_group(rows, **options),
            }
        )
    return {
        "state": "review_required"
        if any(row["state"] == "review_required" for row in streams)
        else "stable"
        if any(row["state"] == "stable" for row in streams)
        else "collecting",
        "streams": streams,
        "method": "Jensen-Shannon effect floor plus two-sample permutation test with Bonferroni correction",
        "interpretation": "A drift signal requests human-reviewed evidence; predictions are never converted to labels.",
        "evaluated_at": time.time(),
    }


class DriftMonitor:
    def __init__(self, maximum_records=5000):
        self.maximum_records = maximum_records

    def evaluate(self):
        with transaction() as session:
            records = list(
                session.scalars(
                    select(Record)
                    .where(Record.kind == "decision")
                    .order_by(Record.timestamp.desc())
                    .limit(self.maximum_records)
                )
            )
            result = analyze_decisions([record.payload for record in reversed(records)])
            row = session.get(Record, "drift-status")
            if row:
                row.payload = result
                row.timestamp = result["evaluated_at"]
            else:
                session.add(Record(id="drift-status", kind="drift_status", payload=result))
        return result

    @staticmethod
    def status():
        with transaction() as session:
            row = session.get(Record, "drift-status")
            return dict(row.payload) if row else {
                "state": "collecting",
                "streams": [],
                "interpretation": "No drift evaluation has run yet.",
            }


drift_monitor = DriftMonitor()
