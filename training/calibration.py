import numpy as np


def probabilities(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(logits, labels):
    labels = np.asarray(labels, dtype=int)
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        raise ValueError("Calibration requires at least 10 labeled samples and two classes")
    candidates = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    losses = [
        -np.log(probabilities(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean()
        for t in candidates
    ]
    return float(candidates[np.argmin(losses)])


def metrics(logits, labels, temperature=1.0):
    labels = np.asarray(labels, dtype=int)
    probs = probabilities(logits, temperature)
    correct = probs.argmax(1) == labels
    confidence = probs.max(1)
    ece = 0.0
    bins = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    one_hot = np.eye(probs.shape[1])[labels]
    coverage = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        keep = confidence >= threshold
        coverage.append(
            {
                "threshold": threshold,
                "coverage": float(keep.mean()),
                "error": float(1 - correct[keep].mean()) if keep.any() else None,
            }
        )
    return {
        "accuracy": float(correct.mean()),
        "ece": float(ece),
        "brier": float(np.square(probs - one_hot).sum(1).mean()),
        "risk_coverage": coverage,
        "samples": len(labels),
    }


def arbitrate(evidence, conflict_threshold=0.25):
    """Only compare calibrated probabilities for the same semantic task."""
    if not evidence or any(not item.get("calibrated") for item in evidence):
        return {"decision": "abstain", "reason": "missing calibration"}
    if len({(e["task"], e["domain"], tuple(e["classes"])) for e in evidence}) != 1:
        return {"decision": "abstain", "reason": "incomparable task/domain/taxonomy"}
    values = np.asarray([item["probabilities"] for item in evidence])
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid probabilities")
    if np.max(np.ptp(values, axis=0)) > conflict_threshold:
        return {"decision": "review", "reason": "calibrated expert disagreement"}
    return {
        "decision": "evidence",
        "probabilities": values.mean(0).tolist(),
        "reason": "compatible calibrated evidence; pooled calibration still requires evaluation",
    }
