"""Compare decoded validation detections through the real ONNX product adapter."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from surveilx.detector_expert import ScratchDetector
from surveilx.onnx_detector import ONNXScratchDetector
from training.detection_pipeline import DetectionImages, collate_detection, digest, evaluate_detections, predict


def validate(run, dataset, export_directory, runtime_path, device_id=1):
    sys.path.insert(0, str(Path(runtime_path).resolve()))
    run, dataset, export_directory = Path(run), Path(dataset), Path(export_directory)
    metadata = json.loads((run / "manifest.json").read_text())
    if digest(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Dataset does not match the frozen model")
    data = json.loads(dataset.read_text())
    samples = [s for s in data["samples"] if s["split"] == "validation"]
    loader = DataLoader(DetectionImages(dataset.parent, samples, metadata["config"]["image_size"]),
                        batch_size=1, collate_fn=collate_detection)
    torch.set_num_threads(1)
    cpu = ScratchDetector(run)
    onnx = ONNXScratchDetector(run, export_directory, ["DmlExecutionProvider", "CPUExecutionProvider"],
                               {"DmlExecutionProvider": {"device_id": str(device_id)}})
    rows = []
    for name, expert in [("pytorch", cpu), ("directml", onnx)]:
        predictions, targets, timing = predict(expert.model, loader, expert.device)
        metrics = evaluate_detections(predictions, targets, len(metadata["classes"]))
        rows.append({"backend": name, "metrics": metrics, "timing": timing})
    result = {"scope": "Complete development validation split; not independent acceptance or test selection",
              "samples": len(samples), "device_id": device_id, "rows": rows,
              "absolute_map50_difference": abs(rows[0]["metrics"]["map50"] - rows[1]["metrics"]["map50"]),
              "deployment_eligible": False}
    (export_directory / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("dataset")
    parser.add_argument("export_directory")
    parser.add_argument("runtime_path")
    parser.add_argument("--device-id", type=int, default=1)
    validate(**vars(parser.parse_args()))
