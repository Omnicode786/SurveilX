"""Summarize completed runs without retraining or choosing models using test scores."""

import json
from pathlib import Path

from training.detection_pipeline import detection_reliability


def summarize(versions, destination):
    rows = []
    dataset = json.loads(Path("data/datasets/pennfudan/manifest.json").read_text())
    targets = [
        {"boxes": s["boxes"], "labels": s["labels"]} for s in dataset["samples"] if s["split"] == "test"
    ]
    for version in versions:
        manifest = json.loads((Path("data/runs") / version / "manifest.json").read_text())
        predictions = json.loads((Path("data/runs") / version / "test_predictions.json").read_text())
        rows.append(
            {
                "version": version,
                "architecture": manifest["architecture"],
                "dataset_sha256": manifest["dataset_sha256"],
                "weights_sha256": manifest["weights_sha256"],
                "parameters": manifest["parameters"],
                "epochs": manifest["config"]["epochs"],
                "metrics": manifest["metrics"],
                "calibration": manifest["calibration"],
                "raw_test_reliability": detection_reliability(predictions, targets),
                "calibrated_test_reliability": detection_reliability(
                    predictions, targets, manifest["calibration"]
                ),
            }
        )
    if len({row["dataset_sha256"] for row in rows}) != 1:
        raise ValueError("Compared models must use the same dataset manifest")
    result = {
        "runs": rows,
        "scope": "Single-seed Penn-Fudan pedestrian detection; 28 test images",
        "scratch_higher_ap50_in_this_run": rows[0]["metrics"]["map50"]
        > max(r["metrics"]["map50"] for r in rows[1:]),
        "limitations": [
            "Unequal initialization: YOLO inherits external pretraining",
            "Different training objectives/augmentation and epochs",
            "No multi-seed confidence interval or independent-domain acceptance",
            "Timing from concurrent training jobs is not a fair speed benchmark",
        ],
    }
    Path(destination).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            summarize(
                ["scratch-pennfudan-v1", "yolo-baseline-pennfudan-v1", "yolo-rai-pennfudan-v1"],
                "reports/detection-comparison.json",
            ),
            indent=2,
        )
    )
