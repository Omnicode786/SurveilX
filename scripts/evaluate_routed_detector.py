"""Read-only native-profile CPU comparison of routed and individual detectors.

Uses existing validation data, not independent acceptance; no deployment changes.
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import torch
from surveilx.detector_expert import AdaptedYOLODetector, ScratchDetector
from surveilx.model_router import CategoryRouter
from surveilx.model_selection import evaluation_for
from training.datasets import digest, validate_manifest
from training.detection_pipeline import evaluate_detections


def as_prediction(found, classes):
    return {
        "boxes": torch.tensor([d.box for d in found]).reshape(-1, 4),
        "labels": torch.tensor([classes.index(d.label) for d in found]).long(),
        "scores": torch.tensor([d.score for d in found]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("versions", nargs="+")
    parser.add_argument("--output", type=Path, help="New report path; preserve earlier comparisons")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    path = Path("data/datasets") / args.dataset / "manifest.json"
    manifest, _ = validate_manifest(path)
    classes = manifest["classes"]
    models = []
    frame_outputs = {}
    for version in args.versions:
        run = Path("data/runs") / version
        metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert metadata["dataset_sha256"] == digest(path) and metadata["classes"] == classes
        expert = (ScratchDetector if metadata["architecture"] == "sva-detector" else AdaptedYOLODetector)(run)
        assert expert.device == "cpu", "This verification must not compete with active CUDA training"
        original = expert.infer

        def recorded(frame, resolution=None, method=original, name=version):
            found = method(frame, resolution)
            frame_outputs[name] = found
            return found

        expert.infer = recorded
        models.append(
            {
                "version": version,
                "manifest": metadata,
                "expert": expert,
                "runtime_device": "cpu",
                "evaluation": evaluation_for(Path("data"), version, metadata),
            }
        )
    router = CategoryRouter(models)
    assert router.routes
    predictions = {v: [] for v in ["routed", *args.versions]}
    truths = []
    start = time.perf_counter()
    for sample in manifest["samples"]:
        if sample["split"] != "validation":
            continue
        frame = cv2.imread(str(path.parent / sample["image"]))
        frame_outputs.clear()
        out = router.infer(frame, None)
        predictions["routed"].append(as_prediction(out, classes))
        for model in models:
            name = model["version"]
            if name not in frame_outputs:
                model["expert"].infer(frame, None)
            predictions[name].append(as_prediction(frame_outputs[name], classes))
        truths.append(
            {
                "boxes": torch.tensor(sample["boxes"]).reshape(-1, 4),
                "labels": torch.tensor(sample["labels"]).long(),
            }
        )
    measured = {}
    for name, pred in predictions.items():
        metrics = evaluate_detections(pred, truths, len(classes), 0.0)
        # Inputs are already thresholded independently by each calibration contract;
        # do not mislabel truncated-score AP as full low-floor AP.
        metrics.pop("map50")
        metrics.pop("map_iou")
        metrics.pop("ap_definition")
        for row in metrics["per_class"]:
            row.pop("ap50")
            row.pop("ap_iou")
        measured[name] = metrics
    result = {
        "task": "detection",
        "split": "validation",
        "samples": len(truths),
        "classes": classes,
        "dataset_sha256": digest(path),
        "routes": router.routes,
        "metrics": measured,
        "elapsed_seconds": time.perf_counter() - start,
        "device": "cpu",
        "selection_and_measurement_share_validation": True,
        "independent_acceptance": False,
        "calibration_scope": "Each component threshold fixed on its calibration split; pooled reliability requires separate evaluation",
        "runtime_promoted": False,
    }
    out = args.output or Path("data/routing-evaluations") / (args.dataset + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Choose a new report path; earlier comparisons are immutable")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    k: value[k] for k in ("precision", "recall", "f1", "false_negatives", "false_positives")
                }
                for name, value in measured.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
