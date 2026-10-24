"""Train the inherited YOLO baseline or an experimental scene/zone adapter variant."""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import math
import os
import hashlib
import tempfile
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
from training.yolo_model import ConditionedBlock, install_adapters, adapter_version as model_adapter_version, upgrade_adapters
from training.yolo_policy import training_policy, bounded_warmup_iterations


@torch.inference_mode()
def native_validation(model, manifest_path, manifest, image_size, preprocessing):
    """Validate the EMA module using the product's geometry/decoder/AP50 contract."""
    from training.detection_geometry import prepare_image, transform_boxes
    from training.image_io import read_detection_image
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    predictions, targets = [], []
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        for sample in manifest["samples"]:
            if sample["split"] != "validation":
                continue
            image, _ = read_detection_image(resolve_asset(manifest_path.parent, sample["image"]), image_size)
            rgb, geometry = prepare_image(image, image_size, preprocessing)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(parameter.device, dtype=parameter.dtype) / 255
            found = non_max_suppression(model(tensor), conf_thres=0.001, iou_thres=0.5,
                multi_label=False, max_det=100)[0].detach().cpu()
            boxes = found[:, :4].clamp(0, image_size) / image_size
            predictions.append({"boxes": torch.from_numpy(transform_boxes(boxes.numpy(), geometry, inverse=True)),
                "scores": found[:, 4], "labels": found[:, 5].long()})
            targets.append({"boxes": torch.tensor(sample["boxes"]).reshape(-1, 4),
                "labels": torch.tensor(sample["labels"], dtype=torch.long)})
    finally:
        model.train(was_training)
    return evaluate_detections(predictions, targets, len(manifest["classes"]))


class NativeSelectionMixin:
    def save_model(self):
        from training.yolo_resources import protected_save
        return protected_save(self, super().save_model)

    def _get_warmup_iterations(self, num_batches):
        count = super()._get_warmup_iterations(num_batches)
        context = getattr(self, "surveilx_context", {})
        return bounded_warmup_iterations(count, num_batches, self.epochs,
            context.get("policy", {}).get("warmup_fraction_cap"))

    def optimizer_step(self):
        super().optimizer_step()
        self.surveilx_step_attempts = getattr(self, "surveilx_step_attempts", 0) + 1

    def validate(self):
        context = getattr(self, "surveilx_context", None)
        if not context:
            return super().validate()
        previous_best = self.best_fitness
        metrics, _ = super().validate()
        started = time.perf_counter()
        selected_model = self.ema.ema if self.ema else self.model
        measured = native_validation(selected_model, context["path"], context["manifest"],
            self.args.imgsz, context["preprocessing"])
        score = measured["map50"]
        if score is None or not math.isfinite(score):
            raise ValueError("Native validation did not produce a finite AP50")
        # The base validator temporarily updates best_fitness using its own mAP95.
        # Restore one consistent native metric before checkpointing/early stopping.
        self.best_fitness = score if previous_best is None else max(previous_best, score)
        self.surveilx_native_metrics = measured
        self.surveilx_native_seconds = time.perf_counter() - started
        return {**metrics, "metrics/native_AP50": score, "metrics/native_precision": measured["precision"],
            "metrics/native_recall": measured["recall"], "metrics/native_F1": measured["f1"]}, score


class NativeDetectionTrainer(NativeSelectionMixin, DetectionTrainer):
    pass


class ConditionedTrainer(NativeSelectionMixin, DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, torch.nn.Module) and any(
            isinstance(block, ConditionedBlock) for block in weights.modules()
        ):
            # Wrapping after loading loses .base and .adapter checkpoint keys.
            model = install_adapters(super().get_model(cfg, None, verbose), model_adapter_version(weights))
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


def bind_native_selection(model, path, manifest, preprocessing, policy, output):
    """Callbacks are configured now; they only execute during an authorized train call."""
    if not callable(getattr(DetectionTrainer, "_get_warmup_iterations", None)):
        raise ValueError("Installed Ultralytics lacks the verified warmup hook; finetune-v2 requires a compatible version")
    def configure(trainer):
        trainer.surveilx_context = {"path": path, "manifest": manifest, "preprocessing": preprocessing, "policy": policy}

    def epoch_start(trainer):
        trainer.surveilx_step_attempts = 0
        trainer.surveilx_lr_ranges = {}

    def batch_end(trainer):
        from training.yolo_resources import check_resources
        check_resources(output)
        for index, group in enumerate(trainer.optimizer.param_groups):
            rate = group["lr"]
            state = trainer.surveilx_lr_ranges.setdefault(str(index), {"name": group.get("param_group", str(index)),
                "minimum": rate, "maximum": rate})
            state.update(minimum=min(state["minimum"], rate), maximum=max(state["maximum"], rate), last=rate)

    def record(trainer):
        if not hasattr(trainer, "surveilx_native_metrics"):
            return
        file = output / "native_history.json"
        rows = json.loads(file.read_text(encoding="utf-8")) if file.exists() else []
        epoch = trainer.epoch + 1
        row = {"epoch": epoch, "validation_map50": trainer.surveilx_native_metrics["map50"],
            "validation_metrics": trainer.surveilx_native_metrics,
            "selection_criterion": "native validation AP50", "native_validation_seconds": trainer.surveilx_native_seconds,
            "elapsed_seconds": time.time() - trainer.train_time_start if hasattr(trainer, "train_time_start") else None,
            "optimizer_step_attempts": trainer.surveilx_step_attempts, "learning_rate_groups": trainer.surveilx_lr_ranges,
            "nominal_effective_batch_size": trainer.args.nbs, "physical_batch_size": trainer.batch_size,
            "accumulation_last": trainer.accumulate}
        if isinstance(trainer.tloss, dict):
            names = {"box_loss": "box_regression", "cls_loss": "classification", "dfl_loss": "distribution_focal"}
            losses = {names.get(key, key): float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                for key, value in trainer.tloss.items()}
        else:
            values = trainer.tloss.detach().cpu().tolist()
            losses = dict(zip(("box_regression", "classification", "distribution_focal"), values, strict=True))
        row["training"] = {"losses": losses}
        row["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
        row["metric_source"] = "SurveilX native validation, no calibration/test fitting"
        rows = [r for r in rows if r["epoch"] != epoch] + [row]
        temp = file.with_suffix(".tmp")
        temp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temp.replace(file)

    for event, callback in (("on_pretrain_routine_start", configure), ("on_train_epoch_start", epoch_start),
                            ("on_train_batch_end", batch_end), ("on_fit_epoch_end", record)):
        model.add_callback(event, callback)


def write_prepared_pixels(image, pixels):
    """Content-address new PNGs, sharing identical bytes without touching old runs."""
    if image.suffix != ".png":
        if not cv2.imwrite(str(image), pixels):
            raise ValueError(f"Could not prepare training image: {image}")
        return
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise ValueError(f"Could not encode training image: {image}")
    payload = encoded.tobytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    from surveilx.config import settings
    shared = settings.data_dir / "prepared-images" / fingerprint[:2] / f"{fingerprint}.png"
    shared.parent.mkdir(parents=True, exist_ok=True)
    if shared.exists():
        if digest(shared) != fingerprint:
            raise ValueError("Prepared image cache integrity mismatch")
    else:
        with tempfile.NamedTemporaryFile(dir=shared.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            os.replace(temporary, shared)
        finally:
            temporary.unlink(missing_ok=True)
    if image.exists():
        raise ValueError("Refusing to overwrite an existing prepared image")
    try:
        os.link(shared, image)
    except OSError:
        shutil.copy2(shared, image)


def prepare_yolo_data(manifest_path, manifest, directory, image_size=None, preprocessing="stretch", lossless=False, decode_mode="full-v1"):
    """Materialize any validated manifest, preserving all four split memberships."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(manifest["samples"]):
        split = sample["split"]
        source = resolve_asset(manifest_path.parent, sample["image"])
        suffix = ".png" if lossless and image_size is not None else source.suffix.lower()
        image = directory / "images" / split / f"{index:06d}{suffix}"
        label = directory / "labels" / split / f"{index:06d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 4)
        if image_size is None:
            shutil.copy2(source, image)
        else:
            from training.image_io import read_detection_image
            pixels, _ = read_detection_image(source, image_size, decode_mode if split == "train" else "full-v1")
            from training.detection_geometry import prepare_image, transform_boxes
            if pixels is None:
                raise ValueError(f"Could not decode training image: {source}")
            pixels, geometry = prepare_image(pixels, image_size, preprocessing)
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            boxes = transform_boxes(boxes, geometry)
            write_prepared_pixels(image, pixels)
        lines = []
        for box, category in zip(boxes, sample["labels"], strict=True):
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


def predict_split(model, manifest_path, manifest, split, image_size, device, preprocessing="stretch"):
    from training.detection_geometry import prepare_image, transform_boxes
    predictions, targets, timings = [], [], []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        # Match scratch model geometry: RGB square input instead of different letterbox distortion.
        frame = cv2.imread(str(resolve_asset(manifest_path.parent, sample["image"])))
        frame, geometry = prepare_image(frame, image_size, preprocessing)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = model.predict(
            frame, imgsz=image_size, device=device, conf=0.001, iou=0.5, max_det=100, verbose=False
        )[0]
        predictions.append(
            {
                "boxes": torch.from_numpy(transform_boxes(result.boxes.xyxyn.cpu().numpy(), geometry, inverse=True)),
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
    preprocessing="stretch",
    training_profile="auto",
    effective_batch_size=None,
    adapter_version="preserve",
    class_weights_power=0.0,
    initialize_partial=None,
    partial_sha256=None,
    partial_plan=None,
    partial_plan_sha256=None,
):
    if initialize_partial and (not initialize_from or resume_checkpoint):
        raise ValueError("Partial weight initialization needs a preserved parent and a new version")
    if not math.isfinite(class_weights_power) or not 0 <= class_weights_power <= .5:
        raise ValueError("Class balancing power must be finite and between 0 and 0.5")
    if adapter_version not in {"preserve", "v1", "local-v2"} or (baseline and adapter_version != "preserve"):
        raise ValueError("Choose a compatible modified-YOLO adapter version")
    if adapter_version != "preserve" and not initialize_from:
        raise ValueError("Explicit architecture upgrade requires a verified modified-YOLO parent")
    if resume_checkpoint and (adapter_version != "preserve" or class_weights_power != 0):
        raise ValueError("Architecture/balancing changes require a new weight-initialized version")
    if preprocessing not in {"stretch", "letterbox"}:
        raise ValueError("Unknown preprocessing contract")
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
    cv2.setNumThreads(max(1, min(threads, 2)))
    device = torch_device()
    started = time.perf_counter()
    initialization = "pretrained"
    partial_initialization = None
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
    if initialize_partial:
        from training.yolo_initialization import verify_partial
        partial_initialization = verify_partial(initialize_partial, partial_sha256, partial_plan,
            partial_plan_sha256, digest(manifest_path), initialization)
        base = partial_initialization["checkpoint"]
    # Legacy crash resumes retain their original prepared geometry.
    if resume_checkpoint and preprocessing != "stretch":
        raise ValueError("Non-legacy YOLO optimizer resume requires a separately verified geometry contract")
    from training.prepared_data import bind_prepared_data, verified_prepared_data
    sources = [Path(__file__).with_name(name) for name in (
        "yolo_pipeline.py", "yolo_policy.py", "image_io.py", "detection_geometry.py", "detection_pipeline.py", "yolo_model.py", "prepared_data.py")]
    policy_file = output / "training_policy.json"
    if resume_checkpoint:
        if not policy_file.exists():
            raise ValueError("Legacy resume policy is unverified; preserve this run and initialize a new version")
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
        if ((training_profile != "auto" and training_profile != policy["version"])
                or (learning_rate is not None and learning_rate != policy["effective_learning_rate"])
                or (effective_batch_size is not None and effective_batch_size != policy["arguments"].get("nbs"))):
            raise ValueError("Optimizer resume requires the original training policy")
        data = verified_prepared_data(output / "dataset", manifest_path, imgsz, preprocessing, policy_file, sources)
    else:
        policy = training_policy(training_profile, bool(initialize_from), learning_rate, epochs, batch, effective_batch_size)
        data = prepare_yolo_data(manifest_path, manifest, output / "dataset", imgsz,
            preprocessing, policy["lossless_prepared_images"], policy["training_decode_mode"])
        (output / "training_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
        bind_prepared_data(output / "dataset", manifest_path, imgsz, preprocessing, policy_file, sources)
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
        if policy["version"] in {"finetune-v2", "finetune-v3"}:
            bind_native_selection(model, manifest_path, manifest, preprocessing, policy, output)
        model.train(resume=True, trainer=NativeDetectionTrainer if baseline else ConditionedTrainer)
    else:
        model = bound_threads(YOLO(base), threads)
        if initialize_partial and [model.model.names[i] for i in range(len(model.model.names))] != manifest["classes"]:
            raise ValueError("Partial checkpoint taxonomy differs from the verified dataset")
        if adapter_version != "preserve":
            upgrade_adapters(model.model, adapter_version)
        if policy["version"] in {"finetune-v2", "finetune-v3"}:
            bind_native_selection(model, manifest_path, manifest, preprocessing, policy, output)
        model.train(
            trainer=NativeDetectionTrainer if baseline else ConditionedTrainer,
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

            max_det=100,
            cls_pw=class_weights_power,
            **({"mosaic": 0, "close_mosaic": 0} | policy["arguments"]),
        )
    # CSV can contain a final evaluation row; it is not another training epoch.
    epochs_completed = model.trainer.epoch + 1
    stopping = {"epochs_completed": epochs_completed, "maximum_epochs": epochs,
        "stopping_reason": "patience" if epochs_completed < epochs else "epoch_limit",
        "best_epoch": getattr(getattr(model.trainer, "stopper", None), "best_epoch", None)}
    shutil.copy2(model.trainer.best, output / "weights.pt")
    model = bound_threads(YOLO(str(output / "weights.pt")), threads)
    validation_predictions, validation_targets, _ = predict_split(
        model, manifest_path, manifest, "validation", imgsz, device, preprocessing
    )
    candidate_validation = evaluate_detections(validation_predictions, validation_targets, len(manifest["classes"]))
    # Preserve class-level evidence even when aggregate selection retains the parent.
    trained_candidate = {"checkpoint": "fit/weights/best.pt",
        "weights_sha256": digest(output / "fit/weights/best.pt"),
        "dataset_sha256": digest(manifest_path), "validation_metrics": candidate_validation,
        "calibrated": False, "deployment_eligible": False,
        "scope": "Pre-parent-selection trained candidate; validation only; fresh calibration/acceptance required"}
    selection = {"candidate_map50": candidate_validation["map50"], "selected": "candidate", "split": "validation"}
    if initialize_from:
        parent_model = bound_threads(YOLO(str(parent / "weights.pt")), threads)
        parent_predictions, parent_targets, _ = predict_split(
            parent_model, manifest_path, manifest, "validation", imgsz, device, preprocessing
        )
        parent_validation = evaluate_detections(
            parent_predictions, parent_targets, len(manifest["classes"])
        )
        selection["parent_map50"] = parent_validation["map50"]
        trained_candidate["parent_validation_metrics"] = parent_validation
        if selection["parent_map50"] >= selection["candidate_map50"]:
            selection["selected"] = "parent"
            shutil.copy2(parent / "weights.pt", output / "weights.pt")
            model = parent_model
    predictions, targets, _ = predict_split(model, manifest_path, manifest, "calibration", imgsz, device, preprocessing)
    calibration = fit_detection_calibration(predictions, targets)
    predictions, targets, timing = predict_split(model, manifest_path, manifest, "test", imgsz, device, preprocessing)
    raw = evaluate_detections(predictions, targets, len(manifest["classes"]))
    metrics = evaluate_detections(
        calibrated_predictions(predictions, calibration),
        targets,
        len(manifest["classes"]),
        calibration["threshold"],
    )
    parent_evaluation = None
    if initialize_from:
        # Size/geometry changes must not be reported as a learned weight gain.
        # Refit only on calibration, then report test AFTER validation selection.
        parent_metrics = metrics
        if selection["selected"] != "parent":
            pp, pt, _ = predict_split(parent_model, manifest_path, manifest, "calibration", imgsz, device, preprocessing)
            parent_calibration = fit_detection_calibration(pp, pt)
            pp, pt, _ = predict_split(parent_model, manifest_path, manifest, "test", imgsz, device, preprocessing)
            parent_metrics = evaluate_detections(calibrated_predictions(pp, parent_calibration), pt,
                len(manifest["classes"]), parent_calibration["threshold"])
        parent_evaluation = {"dataset_sha256": digest(manifest_path), "weights_sha256": initialization,
            "image_size": imgsz, "preprocessing": preprocessing, "metrics": parent_metrics,
            "scope": "Same-profile calibrated test report; never checkpoint selection"}
    (output / "test_predictions.json").write_text(
        json.dumps([{k: v.tolist() for k, v in p.items()} for p in predictions], indent=2), encoding="utf-8"
    )
    (output / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "task": "detection",
        "architecture": "yolo_baseline" if baseline else "yolo_rai",
        "adapter_architecture": None if baseline else model_adapter_version(model.model),
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
                   "training_profile": training_profile, "resolved_training_profile": policy["version"],
                   "effective_batch_size": effective_batch_size,
                   "adapter_version": adapter_version,
                   "class_weights_power": class_weights_power,
                   "initialize_partial": initialize_partial, "partial_sha256": partial_sha256,
                   "partial_plan": partial_plan, "partial_plan_sha256": partial_plan_sha256,
                   "effective_training_policy": policy,
                   "preprocessing": preprocessing,
                   "training_geometry": "legacy_letterbox" if resume_checkpoint and not initialize_from
                   else "square_matches_runtime"},
        "initialization": initialization,
        "partial_initialization": partial_initialization,
        "validation_selection": selection,
        "parent_evaluation": parent_evaluation,
        "trained_candidate": trained_candidate,
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
        "epochs_completed": epochs_completed,
        "stopping": stopping,
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
    parser.add_argument("--preprocessing", choices=("stretch", "letterbox"), default="stretch")
    parser.add_argument("--training-profile", choices=("auto", "legacy-v1", "finetune-v2", "finetune-v3"), default="auto")
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--adapter-version", choices=("preserve", "v1", "local-v2"), default="preserve")
    parser.add_argument("--class-weights-power", type=float, default=0.0)
    parser.add_argument("--initialize-partial")
    parser.add_argument("--partial-sha256")
    parser.add_argument("--partial-plan")
    parser.add_argument("--partial-plan-sha256")
    args = vars(parser.parse_args())
    args.pop("command")
    print(json.dumps(train(**args), indent=2))


if __name__ == "__main__":
    main()
