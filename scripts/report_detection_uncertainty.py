"""Generate a held-out cluster-bootstrap AP50 interval for an existing detector artifact."""

import argparse
import json
from pathlib import Path

import numpy as np

from surveilx.config import settings
from training.datasets import digest, validate_manifest
from training.detection_pipeline import calibrate_scores
from training.uncertainty import cluster_bootstrap_detection


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown(result):
    lines = [
        f"# Detection uncertainty: {result['version']}",
        "",
        f"Dataset: `{result['dataset']}`. Test AP50: {result['point_map50']:.4f}.",
        "",
    ]
    if result["status"] != "estimated":
        lines.extend(
            [
                f"Interval unavailable: {result['clusters']} groups; at least {result['minimum_clusters']} are required.",
                "",
            ]
        )
    else:
        aggregate = result["map50"]
        aggregate_text = (
            f"{aggregate['lower']:.4f}–{aggregate['upper']:.4f}"
            if aggregate.get("lower") is not None else "unavailable (no valid resamples)"
        )
        lines.extend(
            [
                f"95% cluster-bootstrap interval: {aggregate_text} "
                f"from {result['iterations']} resamples across {result['clusters']} recorded groups.",
                "",
                "| Class | Point AP50 | 95% interval |",
                "|---|---:|---:|",
            ]
        )
        points = {item["class_id"]: item["ap50"] for item in result["point_per_class"]}
        for row in result["per_class"]:
            class_id = row["class_id"]
            point = f"{points[class_id]:.4f}" if points[class_id] is not None else "No test labels"
            bounds = f"{row['lower']:.4f}–{row['upper']:.4f}" if row.get("lower") is not None else "Unavailable"
            lines.append(
                f"| {result['classes'][class_id]} | {point} | {bounds} |"
            )
        lines.append("")
        lines.append(result.get("interpretation", ""))
        if aggregate.get("status") == "insufficient_resamples":
            lines.append("Aggregate interval has fewer than 30 valid resamples; insufficient for interpretation.")
    lines.extend(
        [
            f"{result['warning']}. This interval was not used for model selection.",
            "",
        ]
    )
    return "\n".join(lines)


def report(version, dataset, iterations=500, seed=42):
    run = settings.data_dir / "runs" / version
    dataset_path = settings.data_dir / "datasets" / dataset / "manifest.json"
    manifest = read(run / "manifest.json")
    source, _ = validate_manifest(dataset_path)
    if manifest.get("task") != "detection" or manifest["classes"] != source["classes"]:
        raise ValueError("Artifact and detection dataset taxonomies differ")
    if manifest["dataset_sha256"] != digest(dataset_path):
        raise ValueError("Artifact was not evaluated on this dataset manifest")
    samples = [sample for sample in source["samples"] if sample["split"] == "test"]
    rows = read(run / "test_predictions.json")
    if len(rows) != len(samples):
        raise ValueError("Saved test predictions do not match the test split")
    calibration = manifest.get("calibration", {})
    predictions = [
        {
            "boxes": np.asarray(row["boxes"], dtype=float),
            "scores": calibrate_scores(np.asarray(row["scores"], dtype=float), calibration),
            "labels": np.asarray(row["labels"], dtype=int),
        }
        for row in rows
    ]
    targets = [
        {
            "boxes": np.asarray(sample["boxes"], dtype=float),
            "labels": np.asarray(sample["labels"], dtype=int),
        }
        for sample in samples
    ]
    result = cluster_bootstrap_detection(
        predictions,
        targets,
        [sample["group"] for sample in samples],
        len(source["classes"]),
        calibration.get("threshold", 0.25),
        iterations,
        seed,
    )
    result.update(
        version=version,
        dataset=dataset,
        point_map50=manifest["metrics"]["map50"],
        point_per_class=manifest["metrics"]["per_class"],
        classes=source["classes"],
        dataset_sha256=manifest["dataset_sha256"],
        source_independence=source.get("source_independence", False),
        warning="A narrow interval cannot establish site/source independence when dataset provenance lacks those groups",
    )
    output = settings.data_dir.parent / "reports" / f"{version}-uncertainty.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("dataset")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path, result = report(args.version, args.dataset, args.iterations, args.seed)
    print(json.dumps({"path": str(path), "result": result}, indent=2))
