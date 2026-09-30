"""Train the inherited YOLO baseline or an experimental scene/zone adapter variant."""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from surveilx.accelerators import torch_device
from training.detection_pipeline import (
    calibrated_predictions,
    detection_reliability,
    digest,
    evaluate_detections,
    fit_detection_calibration,
    resolve_asset,
    validate_detection_manifest,
)
from training.yolo_model import install_adapters


class ConditionedTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        return install_adapters(super().get_model(cfg, weights, verbose))


def prepare_yolo_data(manifest_path, manifest, directory):
    """Materialize any validated manifest, preserving all four split memberships."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(manifest["samples"]):
        split = sample["split"]
        source = resolve_asset(manifest_path.parent, sample["image"])
        image = directory / "images" / split / f"{index:06d}{source.suffix.lower()}"
        label = directory / "labels" / split / f"{index:06d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image)
        lines = []
        for box, category in zip(sample["boxes"], sample["labels"], strict=True):
            x1, y1, x2, y2 = box
            lines.append(f"{category} {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}")
        label.write_text("\n".join(lines), encoding="utf-8")
    config = {
        "path": str(directory),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": dict(enumerate(manifest["classes"])),
    }
    path = directory / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def predict_split(model, manifest_path, manifest, split, image_size, device):
    predictions, targets, timings = [], [], []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        # Match scratch model geometry: RGB square input instead of different letterbox distortion.
        frame = cv2.imread(str(resolve_asset(manifest_path.parent, sample["image"])))
        frame = cv2.resize(frame, (image_size, image_size))
        result = model.predict(
            frame, imgsz=image_size, device=device, conf=0.001, iou=0.5, max_det=100, verbose=False
        )[0]
        predictions.append(
            {
                "boxes": result.boxes.xyxyn.cpu(),
                "scores": result.boxes.conf.cpu(),
                "labels": result.boxes.cls.long().cpu(),
            }
        )
        targets.append(
            {
                "boxes": torch.tensor(sample["boxes"]).reshape(-1, 4),
                "labels": torch.tensor(sample["labels"], dtype=torch.long),
            }
        )
        timings.append(result.speed)
    return predictions, targets, timings


def train(
    manifest_path,
    output,
    epochs=10,
    imgsz=256,
    batch=8,
    threads=3,
    baseline=False,
    base="yolo11n.pt",
    seed=42,
    resume_checkpoint=None,
):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, counts, hashes = validate_detection_manifest(manifest_path)
    if epochs < 1 or imgsz < 64 or imgsz % 32 or batch < 1:
        raise ValueError("Positive epochs/batch and image size divisible by 32 >=64 are required")
    if output.exists() and any(p.name != "process.log" for p in output.iterdir()) and not resume_checkpoint:
        raise ValueError("Choose an empty version directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    device = torch_device()
    started = time.perf_counter()
    data = prepare_yolo_data(manifest_path, manifest, output / "dataset")
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        expected = (output / "fit" / "weights" / "last.pt").resolve()
        if checkpoint != expected or not checkpoint.is_file() or (output / "manifest.json").exists():
            raise ValueError("Resume checkpoint must be this unfinished run's fit/weights/last.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        arguments = saved.get("train_args", {})
        if any(
            arguments.get(key) != value
            for key, value in {"epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed}.items()
        ):
            raise ValueError("Resume checkpoint training configuration does not match the requested run")
        if Path(arguments.get("data", "")).resolve() != data.resolve():
            raise ValueError("Resume checkpoint belongs to another prepared dataset")
        model = YOLO(str(checkpoint))
        model.train(resume=True, trainer=DetectionTrainer if baseline else ConditionedTrainer)
    else:
        model = YOLO(base)
        model.train(
            trainer=DetectionTrainer if baseline else ConditionedTrainer,
            data=str(data),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=0,
            device=device,
            seed=seed,
            project=str(output),
            name="fit",
            exist_ok=True,
            plots=False,
            amp=False,
            patience=epochs,
            pretrained=True,
            verbose=False,
            mosaic=0,
            close_mosaic=0,
            max_det=100,
        )
    shutil.copy2(model.trainer.best, output / "weights.pt")
    model = YOLO(str(output / "weights.pt"))
    predictions, targets, _ = predict_split(model, manifest_path, manifest, "calibration", imgsz, device)
    calibration = fit_detection_calibration(predictions, targets)
    predictions, targets, timing = predict_split(model, manifest_path, manifest, "test", imgsz, device)
    raw = evaluate_detections(predictions, targets, len(manifest["classes"]))
    metrics = evaluate_detections(
        calibrated_predictions(predictions, calibration),
        targets,
        len(manifest["classes"]),
        calibration["threshold"],
    )
    (output / "test_predictions.json").write_text(
        json.dumps([{k: v.tolist() for k, v in p.items()} for p in predictions], indent=2), encoding="utf-8"
    )
    (output / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "task": "detection",
        "architecture": "yolo_baseline" if baseline else "yolo_rai",
        "name": "YOLO11n baseline" if baseline else "YOLO11n scene/zone adapter",
        "stage": "candidate",
        "classes": manifest["classes"],
        "domain": manifest.get("domain"),
        "capability_ids": manifest.get("capability_ids", []),
        "synthetic": manifest.get("synthetic", False),
        "dataset_counts": counts,
        "dataset_sha256": digest(manifest_path),
        "weights_sha256": digest(output / "weights.pt"),
        "config": {"image_size": imgsz, "epochs": epochs, "batch_size": batch, "seed": seed, "base": base},
        "calibration": calibration,
        "calibrated": calibration["status"] == "fitted",
        "metrics": metrics,
        "uncalibrated_metrics": raw,
        "raw_test_reliability": detection_reliability(predictions, targets),
        "calibrated_test_reliability": detection_reliability(predictions, targets, calibration),
        "prediction_timing": timing,
        "device": device,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "duration_seconds": time.perf_counter() - started,
        "deployment_eligible": False,
        "license": "Ultralytics AGPL-3.0 / Enterprise terms; dataset retains its own terms",
        "context_supervision": "neutral context only; context training not implemented for YOLO batches",
        "gate_reason": "Experimental candidate; independent domain and hardware acceptance required",
        "outperforms_yolo": "not established",
    }
    (output / "manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("manifest_path")
    parser.add_argument("output")
    for name, default in (("epochs", 10), ("imgsz", 256), ("batch", 8), ("threads", 3), ("seed", 42)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--resume-checkpoint")
    args = vars(parser.parse_args())
    args.pop("command")
    print(json.dumps(train(**args), indent=2))


if __name__ == "__main__":
    main()
