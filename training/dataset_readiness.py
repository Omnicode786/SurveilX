"""Read-only integrity and supervision coverage, never a model quality approval."""
from collections import Counter

from training.datasets import SPLITS, digest, validate_manifest


def inspect_dataset(path):
    manifest, counts = validate_manifest(path)
    task = manifest.get("task", "event")
    labels = {split: Counter() for split in SPLITS}
    images = {split: Counter() for split in SPLITS}
    groups = {split: set() for split in SPLITS}
    empty = Counter()
    for sample in manifest["samples"]:
        split = sample["split"]
        categories = sample["labels"] if task == "detection" else [sample["label"]]
        labels[split].update(categories)
        images[split].update(set(categories))
        groups[split].add(sample["group"])
        empty[split] += not categories
    per_class, issues = [], []
    for index, name in enumerate(manifest["classes"]):
        support = {split: labels[split][index] for split in SPLITS}
        per_class.append({"label": name, "labeled_instances": support,
                          "samples_with_label": {split: images[split][index] for split in SPLITS}})
        for split in SPLITS:
            if not support[split]:
                issues.append({"label": name, "split": split, "kind": "missing_supervision",
                               "message": f"No {split} labels for {name}; this class cannot be learned or measured in that split."})
            elif support[split] < 10:
                issues.append({"label": name, "split": split, "kind": "sparse_supervision",
                               "message": f"Only {support[split]} {split} labels for {name}; fewer than ten is a review warning, not a statistical acceptance threshold."})
    return {"valid": True, "manifest_sha256": digest(path), "task": task, "counts": counts,
            "count_unit": "objects" if task == "detection" else "clips",
            "empty_samples": {split: empty[split] for split in SPLITS},
            "recorded_groups": {split: len(groups[split]) for split in SPLITS},
            "per_class": per_class, "issues": issues,
            "synthetic": bool(manifest.get("synthetic", False)), "provenance": manifest.get("provenance", {}),
            "scope": "Dataset integrity and label coverage only. Group identifiers do not prove source independence; model training, evaluation and acceptance are separate.",
            "model_quality_verified": False}
