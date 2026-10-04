"""Train the inherited YOLO baseline or an experimental scene/zone adapter variant."""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import math
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
from training.yolo_model import ConditionedBlock, install_adapters


class ConditionedTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, torch.nn.Module) and any(
            isinstance(block, ConditionedBlock) for block in weights.modules()
        ):
            # Wrapping after loading loses .base and .adapter checkpoint keys.
            model = install_adapters(super().get_model(cfg, None, verbose))
            model.load(weights, verbose=verbose)
            return model
        return install_adapters(super().get_model(cfg, weights, verbose))


def bound_threads(model, threads):
    # Ultralytics device setup resets PyTorch's thread count; enforce our CPU budget afterwards.
    def apply_budget(_):
        torch.set_num_threads(threads)

    model.add_callback("on_pretrain_routine_start", apply_budget)
    model.add_callback("on_predict_start", apply_budget)
    return model


def prepare_yolo_data(manifest_path, manifest, directory, image_size=None):
    """Materialize any validated manifest, preserving all four split memberships."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(manifest["samples"]):
        split = sample["split"]
        source = resolve_asset(manifest_path.parent, sample["image"])
        image = directory / "images" / split / f"{index:06d}{source.suffix.lower()}"
        label = directory / "labels" / split / f"{index:06d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if image_size is None:
            shutil.copy2(source, image)
        else:
            pixels = cv2.imread(str(source))
            if pixels is None or not cv2.imwrite(str(image), cv2.resize(pixels, (image_size, image_size))):
                raise ValueError(f"Could not prepare training image: {source}")
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
    initialize_from=None,
    learning_rate=None,
    patience=None,
):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, counts, hashes = validate_detection_manifest(manifest_path)
    if epochs < 1 or imgsz < 64 or imgsz % 32 or batch < 1:
        raise ValueError("Positive epochs/batch and image size divisible by 32 >=64 are required")
    if threads < 1 or (learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0)):
        raise ValueError("Threads and optional learning rate must be positive")
    if patience is not None and patience < 1:
        raise ValueError("Optional patience must be positive")
    if output.exists() and any(p.name != "process.log" for p in output.iterdir()) and not resume_checkpoint:
        raise ValueError("Choose an empty version directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    device = torch_device()
    started = time.perf_counter()
    initialization = "pretrained"
    if initialize_from:
        parent = Path(initialize_from).resolve()
        metadata = json.loads((parent / "manifest.json").read_text())
        expected_architecture = "yolo_baseline" if baseline else "yolo_rai"
        if (metadata.get("task") != "detection" or metadata.get("classes") != manifest["classes"]
                or metadata.get("domain") != manifest.get("domain")
                or metadata.get("architecture") != expected_architecture):
            raise ValueError("Continuation requires identical task, domain, taxonomy and YOLO architecture")
        if digest(parent / "weights.pt") != metadata.get("weights_sha256"):
            raise ValueError("Continuation checkpoint checksum mismatch")
        base = str(parent / "weights.pt")
        initialization = metadata["weights_sha256"]
    # Legacy crash resumes retain their original prepared geometry.
    data = prepare_yolo_data(manifest_path, manifest, output / "dataset", None if resume_checkpoint else imgsz)
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
        model = bound_threads(YOLO(str(checkpoint)), threads)
        model.train(resume=True, trainer=DetectionTrainer if baseline else ConditionedTrainer)
    else:
        model = bound_threads(YOLO(base), threads)
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
            patience=epochs if patience is None else patience,
            pretrained=True,
            verbose=False,
            mosaic=0,
            close_mosaic=0,
            max_det=100,
            **({"optimizer": "AdamW", "lr0": learning_rate, "cos_lr": True} if learning_rate else {}),
        )
    shutil.copy2(model.trainer.best, output / "weights.pt")
    model = bound_threads(YOLO(str(output / "weights.pt")), threads)
    validation_predictions, validation_targets, _ = predict_split(
        model, manifest_path, manifest, "validation", imgsz, device
    )
    selection = {"candidate_map50": evaluate_detections(
        validation_predictions, validation_targets, len(manifest["classes"])
    )["map50"], "selected": "candidate", "split": "validation"}
    if initialize_from:
        parent_model = bound_threads(YOLO(str(parent / "weights.pt")), threads)
        parent_predictions, parent_targets, _ = predict_split(
            parent_model, manifest_path, manifest, "validation", imgsz, device
        )
        selection["parent_map50"] = evaluate_detections(
            parent_predictions, parent_targets, len(manifest["classes"])
        )["map50"]
        if selection["parent_map50"] >= selection["candidate_map50"]:
            selection["selected"] = "parent"
            shutil.copy2(parent / "weights.pt", output / "weights.pt")
            model = parent_model
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
        "config": {"image_size": imgsz, "epochs": epochs, "batch_size": batch, "seed": seed,
                   "base": base, "learning_rate": learning_rate, "patience": patience,
                   "threads": threads,
                   "training_geometry": "legacy_letterbox" if resume_checkpoint and not initialize_from
                   else "square_matches_runtime"},
        "initialization": initialization,
        "validation_selection": selection,
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
    parser.add_argument("--initialize-from")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    args = vars(parser.parse_args())
    args.pop("command")
    print(json.dumps(train(**args), indent=2))


if __name__ == "__main__":
    main()
