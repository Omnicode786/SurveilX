"""Report class coverage and training object sizes without changing frozen data."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_accuracy_campaign import coverage, paths, read


def audit(plan_path, output):
    plan = read(Path(plan_path))
    datasets = []
    seen = set()
    for job in plan["jobs"]:
        if job["dataset"] in seen:
            continue
        seen.add(job["dataset"])
        path, _, _ = paths(job)
        manifest = read(path)
        classes = coverage(manifest)
        for index, row in enumerate(classes):
            sizes = [256 * min(box[2] - box[0], box[3] - box[1])
                     for sample in manifest["samples"] if sample["split"] == "train"
                     for box, label in zip(sample.get("boxes", []), sample.get("labels", []), strict=True)
                     if label == index]
            row["training_short_side_pixels_at_256"] = {
                "median": float(np.median(sizes)) if sizes else None,
                "fraction_below_stride_8": float(np.mean(np.array(sizes) < 8)) if sizes else None,
            }
            row["missing_splits"] = [split for split, count in row["labeled_instances"].items() if count == 0]
        datasets.append({"dataset": job["dataset"], "classes": classes, "synthetic": manifest.get("synthetic", False),
                         "provenance": manifest.get("provenance", {})})
    report = {"scope": "Label counts describe coverage, not statistical independence or sufficient sample size",
              "datasets": datasets}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Accuracy data audit", "", report["scope"], "",
             "Object sizes use training labels only. Sub-stride objects flag a resolution constraint; this is not a measured detection limit.", ""]
    for dataset in datasets:
        lines += [f"## {dataset['dataset']}", "", "| Class | Train | Validation | Calibration | Test | Median short side at 256 px |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in dataset["classes"]:
            counts = row["labeled_instances"]
            size = row["training_short_side_pixels_at_256"]["median"]
            lines.append(f"| {row['class']} | {counts['train']} | {counts['validation']} | {counts['calibration']} | {counts['test']} | "
                         + (f"{size:.1f}" if size is not None else "—") + " |")
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("output")
    args = parser.parse_args()
    audit(args.plan, args.output)
