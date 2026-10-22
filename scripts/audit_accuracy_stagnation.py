"""Read-only g7 schedule and candidate-vs-parent validation audit. Never trains."""

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from training.datasets import digest
from training.yolo_pipeline import native_validation


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve previous diagnostics; choose a new report")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    plan = json.loads(Path("data/generations/accuracy-routing-g7-1015/plan.json").read_text(encoding="utf-8"))
    results = []
    for job in plan["jobs"]:
        if job["pipeline"] != "yolo":
            continue
        run = Path("data/runs") / job["version"]
        arguments = yaml.safe_load((run / "fit/args.yaml").read_text(encoding="utf-8"))
        with (run / "fit/results.csv").open(encoding="utf-8") as stream:
            curve = list(csv.DictReader(stream))
        row = {"version": job["version"], "task": "detection", "split": "validation",
            "dataset_sha256": job["dataset_sha256"], "arguments": arguments,
            "warmup_bias_to_requested_lr_ratio": arguments["warmup_bias_lr"] / arguments["lr0"],
            "first_epoch_lr_groups": {k: v for k, v in curve[0].items() if k.startswith("lr/")},
            "nominal_accumulation_after_warmup": max(1, round(arguments["nbs"] / arguments["batch"])),
            "library_best_ap50_epoch": max(curve, key=lambda r: float(r["metrics/mAP50(B)"]))["epoch"],
            "library_best_map95_epoch": max(curve, key=lambda r: float(r["metrics/mAP50-95(B)"]))["epoch"]}
        dataset = Path("data/datasets") / job["dataset"] / "manifest.json"
        assert digest(dataset) == job["dataset_sha256"]
        manifest = json.loads(dataset.read_text(encoding="utf-8"))
        for label, checkpoint in (("parent", Path("data/runs") / job["parent"] / "weights.pt"),
                                  ("trained_candidate", run / "fit/weights/best.pt")):
            fingerprint = digest(checkpoint)
            model = YOLO(str(checkpoint)).model.float().eval()
            assert list(model.names.values()) == manifest["classes"]
            measured = native_validation(model, dataset, manifest, job["options"]["imgsz"],
                job["options"].get("preprocessing", "stretch"))
            assert digest(checkpoint) == fingerprint
            row[label] = {"checkpoint_sha256": fingerprint, "metrics": measured}
            del model
            gc.collect()
        row["per_class_changes"] = [{"class": name,
            "support": row["parent"]["metrics"]["per_class"][i]["ground_truth"],
            "parent_ap50": row["parent"]["metrics"]["per_class"][i]["ap50"],
            "candidate_ap50": row["trained_candidate"]["metrics"]["per_class"][i]["ap50"]}
            for i, name in enumerate(manifest["classes"])]
        row["estimated_batches_per_epoch"] = math.ceil(sum(s["split"] == "train" for s in manifest["samples"]) / arguments["batch"])
        results.append(row)
        print(json.dumps({"version": job["version"], "parent": row["parent"]["metrics"]["map50"],
            "candidate": row["trained_candidate"]["metrics"]["map50"]}), flush=True)
    output.write_text(json.dumps({"scope": "Read-only full validation, equal geometry; no optimizer, calibration fit, test selection or activation",
        "evaluations": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    audit(parser.parse_args().output)
