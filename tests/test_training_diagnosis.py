import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from training.image_io import read_detection_image
from training.yolo_policy import training_policy, bounded_warmup_iterations


def test_continuation_does_not_inherit_large_bias_lr_or_64_image_accumulation():
    for rate in (0.00005, 0.0001):
        policy = training_policy("auto", True, rate, 4, 4)
        args = policy["arguments"]
        assert args["warmup_bias_lr"] == 0 and args["lr0"] == rate
        assert args["nbs"] == 8 and args["scale"] == 0.15
        assert bounded_warmup_iterations(100, 10, 4, policy["warmup_fraction_cap"]) == 4
        assert policy["lossless_prepared_images"]
    assert training_policy("auto", False, 1e-4, 4, 4)["arguments"]["warmup_bias_lr"] == 0
    assert training_policy("auto", True, None, 8, 4)["effective_learning_rate"] == 1e-4
    assert training_policy("legacy-v1", True, 1e-4, 8, 4)["warmup_fraction_cap"] is None
    with pytest.raises(ValueError):
        training_policy("auto", True, 1e-4, 8, 4, effective_batch_size=2)
    with pytest.raises(ValueError):
        training_policy("finetune-v2", True, 1e-4, 8, 4, resume=True)


def test_native_fitness_never_compares_ap50_with_library_map95(monkeypatch):
    import training.yolo_pipeline as pipeline

    trainer = object.__new__(pipeline.ConditionedTrainer)
    trainer.best_fitness = 0.5
    trainer.ema = SimpleNamespace(ema=torch.nn.Identity())
    trainer.args = SimpleNamespace(imgsz=64)
    trainer.surveilx_context = {"path": Path("unused"), "manifest": {}, "preprocessing": "stretch"}

    def library(self):
        self.best_fitness = 0.9
        return {"metrics/mAP50-95(B)": 0.9}, 0.9

    monkeypatch.setattr(pipeline.DetectionTrainer, "validate", library)
    monkeypatch.setattr(pipeline, "native_validation", lambda *a: {"map50": 0.4, "precision": 0.3, "recall": 0.2, "f1": 0.24})
    metrics, score = trainer.validate()
    assert score == 0.4 and trainer.best_fitness == 0.5
    assert metrics["metrics/native_AP50"] == 0.4


def test_reduced_jpeg_decode_preserves_boxes_and_bounds_memory(tmp_path):
    from training.detection_pipeline import DetectionImages

    source = np.full((2000, 3000, 3), (20, 80, 210), np.uint8)
    path = tmp_path / "large.jpg"
    assert cv2.imwrite(str(path), source)
    frame, info = read_detection_image(path, 256, "bounded-jpeg-v1")
    assert info["factor"] == 2 and frame.shape == (1000, 1500, 3)
    assert info["decoded_bytes"] == info["source_bgr_bytes"] // 4
    sample = {"image": path.name, "boxes": [[0.1, 0.2, 0.9, 0.8]], "labels": [0]}
    plain = DetectionImages(tmp_path, [sample], 256)[0]
    reduced = DetectionImages(tmp_path, [sample], 256, decode_mode="bounded-jpeg-v1")[0]
    assert torch.equal(plain[3]["boxes"], reduced[3]["boxes"])
    assert reduced[3]["image_decode"]["factor"] == 2
    # A narrow image must retain its short axis.
    narrow = tmp_path / "narrow.jpg"
    cv2.imwrite(str(narrow), source[:200])
    assert read_detection_image(narrow, 256, "bounded-jpeg-v1")[1]["factor"] == 1


def test_decode_retry_uses_same_file_and_never_skips_failure(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.full((64, 96, 3), 40, np.uint8))
    expected = cv2.imread(str(path))
    monkeypatch.setattr(cv2, "imread", lambda *a: None)
    actual, info = read_detection_image(path, 64)
    assert info["fallback"] and np.array_equal(actual, expected)
    monkeypatch.setattr(cv2, "imdecode", lambda *a: None)
    with pytest.raises(ValueError, match="Cannot decode image"):
        read_detection_image(path, 64)


def test_lossless_prepared_validation_matches_runtime_pixels(tmp_path):
    from training.yolo_pipeline import prepare_yolo_data
    from training.detection_geometry import prepare_image

    source = np.random.default_rng(3).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    path = tmp_path / "original.jpg"
    cv2.imwrite(str(path), source)
    manifest = {"classes": ["person"], "samples": [{"image": path.name, "split": "validation", "boxes": [], "labels": []}]}
    prepare_yolo_data(tmp_path / "manifest.json", manifest, tmp_path / "prepared", 96, "letterbox", lossless=True)
    expected, _ = prepare_image(cv2.imread(str(path)), 96, "letterbox")
    stored = cv2.imread(str(tmp_path / "prepared/images/validation/000000.png"))
    assert np.array_equal(stored, cv2.cvtColor(expected, cv2.COLOR_RGB2BGR))


def test_native_validator_matches_real_checkpoint_runtime_without_training(monkeypatch):
    import training.yolo_pipeline as pipeline
    from ultralytics import YOLO

    run = Path(__file__).resolve().parents[1] / "data/runs/cuda-g4-weapons-yolo"
    if not (run / "weights.pt").exists():
        pytest.skip("Local immutable checkpoint not distributed with source")
    root = run.parents[1] / "datasets/dangerous-items-development-v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sample = next(s for s in manifest["samples"] if s["split"] == "validation")
    small = {**manifest, "samples": [sample]}
    torch.set_num_threads(2)
    model = YOLO(str(run / "weights.pt"))
    captured = []
    original = pipeline.evaluate_detections

    def record(predictions, targets, count):
        captured.extend(copy.deepcopy(predictions))
        return original(predictions, targets, count)

    monkeypatch.setattr(pipeline, "evaluate_detections", record)
    measured = pipeline.native_validation(model.model, root / "manifest.json", small, 384, "stretch")
    predictions, truths, _ = pipeline.predict_split(model, root / "manifest.json", small, "validation", 384, "cpu")
    assert len(captured) == len(predictions) == 1
    assert torch.equal(captured[0]["labels"], predictions[0]["labels"])
    for key in ("boxes", "scores"):
        assert torch.allclose(captured[0][key], predictions[0][key], atol=1e-4, rtol=1e-4)
    assert measured["map50"] == pytest.approx(original(predictions, truths, len(manifest["classes"]))["map50"])


def test_v3_telemetry_preserves_old_loss_and_classifier_gradient():
    from training.detector_model import SVADetector
    from training.detector_v3 import v3_detection_loss
    from training.observability import EpochLossSummary

    torch.set_num_threads(2)
    path = Path(__file__).resolve().parents[1] / "data/generations/accuracy-routing-g7-1015/source_snapshot/training/detector_v3.py"
    if not path.exists():
        pytest.skip("Local frozen historical source not distributed")
    old_loss = runpy.run_path(str(path))["v3_detection_loss"]
    model = SVADetector(classes=4, width=8, depth=1, architecture_version=3).eval()
    heads = model(torch.rand(1, 3, 64, 64))
    targets = [{"boxes": torch.tensor([[0.2, 0.1, 0.7, 0.8]]), "labels": torch.tensor([2])}]
    before = old_loss(heads, targets)
    after = v3_detection_loss(heads, targets)
    for key in ("loss", "objectness", "classification", "box_giou", "box_l1"):
        assert torch.equal(before[key], after[key])
    parameter = next(model.parameters())
    assert torch.equal(torch.autograd.grad(before["loss"], parameter, retain_graph=True)[0], torch.autograd.grad(after["loss"], parameter)[0])
    summary = EpochLossSummary()
    summary.update(after, 1)
    assert sum(summary.report()["score_quality"]["per_class_positive_counts"]) == int(after["positives"])


def test_unpromoted_candidate_requires_fingerprints_and_never_enters_routing(tmp_path):
    from surveilx.model_selection import catalog
    from training.datasets import digest

    run = tmp_path / "runs" / "candidate"
    (run / "fit/weights").mkdir(parents=True)
    (run / "weights.pt").write_bytes(b"selected parent")
    (run / "fit/weights/best.pt").write_bytes(b"trained candidate")
    metrics = {"per_class": [{"class_id": 0, "ground_truth": 20, "ap50": 0.5}]}
    manifest = {"task": "detection", "classes": ["smoke"], "weights_sha256": digest(run / "weights.pt"),
        "dataset_sha256": "dataset", "trained_candidate": {"checkpoint": "fit/weights/best.pt",
            "weights_sha256": digest(run / "fit/weights/best.pt"), "dataset_sha256": "dataset",
            "validation_metrics": metrics,
            "parent_validation_metrics": {"per_class": [{"class_id": 0, "ap50": 0.3}]}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    report = catalog(tmp_path)
    assert not report["recommendations"]
    candidate = report["unpromoted_candidates"][0]
    assert candidate["calibrated"] is False and candidate["deployment_eligible"] is False
    assert candidate["class_improvements"][0]["delta"] == pytest.approx(0.2)
    (run / "fit/weights/best.pt").write_bytes(b"changed candidate")
    assert not catalog(tmp_path)["unpromoted_candidates"]


def test_prepared_resume_checks_assets_without_rewriting_them(tmp_path):
    from training.prepared_data import bind_prepared_data, verified_prepared_data

    prepared = tmp_path / "prepared"
    (prepared / "images/train").mkdir(parents=True)
    (prepared / "labels/train").mkdir(parents=True)
    image = prepared / "images/train/one.png"
    image.write_bytes(b"prepared image")
    label = prepared / "labels/train/one.txt"
    label.write_text("0 .5 .5 .1 .1")
    (prepared / "dataset.yaml").write_text("names: [person]")
    manifest, policy, source = [tmp_path / name for name in ("manifest.json", "policy.json", "source.py")]
    for file in (manifest, policy, source):
        file.write_text("original")
    args = (prepared, manifest, 256, "stretch", policy, [source])
    with pytest.raises(ValueError, match="Legacy prepared"):
        verified_prepared_data(*args)
    assert image.read_bytes() == b"prepared image"
    bind_prepared_data(*args)
    assert verified_prepared_data(*args) == prepared / "dataset.yaml"
    for file in (image, label, source, policy, manifest):
        before = file.read_bytes()
        file.write_bytes(b"changed")
        with pytest.raises(ValueError, match="resume rejected"):
            verified_prepared_data(*args)
        assert file.read_bytes() == b"changed"  # Verification cannot repair/reprepare it.
        file.write_bytes(before)
    with pytest.raises(ValueError, match="resume rejected"):
        verified_prepared_data(prepared, manifest, 512, "stretch", policy, [source])
    with pytest.raises(ValueError, match="resume rejected"):
        verified_prepared_data(prepared, manifest, 256, "letterbox", policy, [source])
    (prepared / "dataset.yaml").unlink()
    with pytest.raises(ValueError, match="dataset.yaml is missing"):
        verified_prepared_data(*args)


@pytest.mark.parametrize("loss_dict", [False, True])
def test_native_history_records_real_lr_ranges_steps_and_per_class_metrics(tmp_path, loss_dict):
    from training.yolo_pipeline import bind_native_selection

    callbacks = {}
    model = SimpleNamespace(add_callback=lambda event, callback: callbacks.update({event: callback}))
    bind_native_selection(model, tmp_path / "manifest.json", {}, "stretch", {"version": "finetune-v2"}, tmp_path)
    trainer = SimpleNamespace(epoch=0, args=SimpleNamespace(nbs=8), batch_size=4, accumulate=2,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-5}, {"lr": 2e-5}]),
        tloss=torch.tensor([1., 2., 3.]), surveilx_native_seconds=0.2,
        surveilx_native_metrics={"map50": 0.4, "per_class": [{"class_id": 0, "ap50": 0.4}]})
    if loss_dict:
        trainer.tloss = {"box_loss": torch.tensor(1.), "cls_loss": torch.tensor(2.), "dfl_loss": torch.tensor(3.)}
    callbacks["on_pretrain_routine_start"](trainer)
    callbacks["on_train_epoch_start"](trainer)
    callbacks["on_train_batch_end"](trainer)
    trainer.optimizer.param_groups[0]["lr"] = 1e-4
    trainer.surveilx_step_attempts = 3  # No real optimizer is constructed or stepped.
    callbacks["on_train_batch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)  # Idempotent final callback.
    rows = json.loads((tmp_path / "native_history.json").read_text())
    assert len(rows) == 1 and rows[0]["optimizer_step_attempts"] == 3
    assert rows[0]["learning_rate_groups"]["0"] == {"name": "0", "minimum": 1e-5, "maximum": 1e-4, "last": 1e-4}
    assert rows[0]["validation_metrics"]["per_class"][0]["ap50"] == 0.4
    assert rows[0]["training"]["losses"] == {"box_regression": 1., "classification": 2., "distribution_focal": 3.}
    from scripts.watch_training import format_epoch
    rendered = "\n".join(format_epoch(rows[0]))
    assert "group 0 1e-05..0.0001" in rendered and "optimizer step attempts 3" in rendered
