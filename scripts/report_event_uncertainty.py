"""Describe uncertainty of saved event predictions without training or test selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from surveilx.config import settings
from training.datasets import digest, validate_manifest
from training.event_uncertainty import event_uncertainty


def report(version, dataset):
    root = settings.data_dir / "runs" / version
    source = settings.data_dir / "datasets" / dataset / "manifest.json"
    metadata = json.loads((root / "manifest.json").read_text())
    manifest, _ = validate_manifest(source)
    if (
        metadata.get("task") != "event"
        or metadata["dataset_sha256"] != digest(source)
        or metadata["classes"] != manifest["classes"]
        or metadata["weights_sha256"] != digest(root / "weights.pt")
    ):
        raise ValueError("Frozen event artifact and dataset do not match")
    predictions = root / "heldout_predictions.npz"
    checksum = digest(predictions)
    if metadata.get("heldout_predictions_sha256", checksum) != checksum:
        raise ValueError("Saved predictions checksum mismatch")
    samples = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    with np.load(predictions, allow_pickle=False) as data:
        logits, labels = data["logits"], data["labels"]
    if not np.array_equal(labels, [sample["label"] for sample in samples]):
        raise ValueError("Saved test labels differ from the frozen manifest")
    accuracy = float((logits.argmax(1) == labels).mean())
    if abs(accuracy - metadata["metrics"]["accuracy"]) > 1e-9:
        raise ValueError("Saved predictions disagree with the recorded metric")
    result = event_uncertainty(logits, labels, [sample["group"] for sample in samples])
    result.update(
        version=version,
        dataset=dataset,
        accuracy=accuracy,
        classes=manifest["classes"],
        dataset_sha256=digest(source),
        heldout_predictions_sha256=checksum,
        predictions_hash_recorded_at_training="heldout_predictions_sha256" in metadata,
        source_independence_verified=manifest.get("provenance", {}).get(
            "source_independence_verified", False
        ),
    )
    target = Path("reports") / f"{version}-event-uncertainty.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(report(args.version, args.dataset))
