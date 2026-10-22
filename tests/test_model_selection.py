import copy
from types import SimpleNamespace
import pytest
import torch
from surveilx.model_selection import rank_categories
from surveilx.model_router import CategoryRouter
from surveilx.vision import Detection
from training.checkpointing import (
    save_training_state,
    load_training_state,
    restore_rng,
    step_accumulated,
)
from training.datasets import digest


def candidate(version, ap=0.6, recall=0.6, dataset="same"):
    return {
        "version": version,
        "stage": "candidate",
        "manifest": {
            "task": "detection",
            "domain": "office",
            "synthetic": False,
            "dataset_sha256": dataset,
            "classes": ["person", "helmet"],
            "calibrated": True,
            "weights_sha256": version,
            "calibration": {"status": "fitted"},
        },
        "evaluation": {
            "split": "validation",
            "metrics": {
                "per_class": [
                    {"class_id": 0, "ground_truth": 10, "ap50": ap, "recall": recall, "f1": recall},
                    {"class_id": 1, "ground_truth": 4, "ap50": 0.4, "recall": 0.5, "f1": 0.5},
                ]
            },
        },
    }


def test_category_rank_contracts_recall_and_approval():
    strong_ap = candidate("a", 0.8, 0.1)
    balanced = candidate("b", 0.7, 0.8)
    selected = rank_categories([strong_ap, balanced])
    assert next(r for r in selected if r["category"] == "person")["selected"]["version"] == "b"
    assert not selected[0]["production_approved"]
    other = candidate("c", 1, 1, "different")
    reports = rank_categories([balanced, other])
    assert len(reports) == 4
    bad = candidate("bad")
    bad["evaluation"]["split"] = "test"
    assert not rank_categories([bad])
    bad = candidate("bad")
    bad["manifest"]["calibration"]["status"] = "missing"
    assert not rank_categories([bad])
    bad["manifest"]["calibration"] = None
    assert not rank_categories([bad])


def test_router_deduplicates_recovers_and_isolates_classes():
    calls = []
    a = candidate("a", 0.9, 0.9)
    b = candidate("b", 0.5, 0.5)
    for model in (a, b):
        version = model["version"]
        model["expert"] = SimpleNamespace(
            infer=lambda *args, v=version: (
                calls.append(v)
                or [Detection([0, 0, 1, 1], "person", 0.9), Detection([0, 0, 0.2, 0.2], "helmet", 0.8)]
            )
        )
    router = CategoryRouter([a, b])
    result = router.infer(None)
    assert calls == ["a"] and sorted(d.label for d in result) == ["helmet", "person"]

    def fail(*args):
        raise RuntimeError("device unavailable")

    a["expert"].infer = fail
    router.infer(None)
    assert router.failed_until["a"] > 0 and "b" in calls
    router.failed_until["a"] = 0
    a["expert"].infer = lambda *args: []
    assert router.infer(None) == []
    different = candidate("different", dataset="different")
    different["expert"] = b["expert"]
    assert CategoryRouter([a, different]).routes == []


def test_checkpoint_integrity_rng_and_accumulation(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for sample in xs:
        model(sample).square().mean().backward()
    step_accumulated(optimizer, model, 3)
    reference(xs).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0, error_if_nonfinite=True)
    reference_optimizer.step()
    assert all(torch.allclose(p, q, atol=1e-7) for p, q in zip(model.parameters(), reference.parameters()))
    scheduler.step()
    path = tmp_path / "last.training.pt"
    recipe = {"dataset": "fixed"}
    save_training_state(path, model, optimizer, scheduler, 1, recipe, {}, [])
    saved = load_training_state(path, digest(path), recipe)
    expected = torch.rand(3)
    restore_rng(saved)
    assert torch.equal(torch.rand(3), expected)
    with pytest.raises(ValueError):
        load_training_state(path, "corrupt", recipe)
    with pytest.raises(ValueError):
        load_training_state(path, digest(path), {"dataset": "moved"})
    assert not path.with_suffix(".pt.tmp").exists()


def test_model_performance_api_auth_and_no_activation(client):
    from test_api import login

    assert client.get("/api/model-performance").status_code == 401
    login(client)
    report = client.get("/api/model-performance")
    assert report.status_code == 200
    assert report.json()["automatic_activation"] is False and report.json()["models"] == []


def test_detection_optimizer_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import json
    import cv2
    import numpy as np
    from training.detection_pipeline import train

    samples = []
    for i, split in enumerate(
        ("train", "train", "validation", "validation", "calibration", "calibration", "test", "test")
    ):
        name = f"{i}.png"
        cv2.imwrite(
            str(tmp_path / name), np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        samples.append(
            {"image": name, "split": split, "group": name, "boxes": [[0.2, 0.2, 0.8, 0.8]], "labels": [0]}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "detection",
                "domain": "unit-correctness",
                "license": "test-fixture",
                "synthetic": True,
                "classes": ["person"],
                "samples": samples,
            }
        )
    )
    kwargs = dict(
        epochs=2,
        image_size=64,
        width=8,
        depth=1,
        batch_size=1,
        threads=2,
        architecture_version=3,
        accumulation_steps=2,
        seed=73,
    )
    full = tmp_path / "full"
    train(manifest, full, **kwargs)
    original = torch.optim.AdamW.step
    calls = 0

    def interrupt(self, *args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(self, *args, **kw)

    partial = tmp_path / "interrupted"
    with monkeypatch.context() as context:
        context.setattr(torch.optim.AdamW, "step", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(manifest, partial, **kwargs)
    checkpoint = partial / "last.training.pt"
    resumed = tmp_path / "resumed"
    train(manifest, resumed, resume_from=checkpoint, resume_sha256=digest(checkpoint), **kwargs)
    expected = torch.load(full / "weights.pt", weights_only=True)
    actual = torch.load(resumed / "weights.pt", weights_only=True)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert len(json.loads((resumed / "history.json").read_text())) == 2


def test_yolo_letterbox_geometry_and_channel_parity(tmp_path):
    import cv2
    import numpy as np
    from training.yolo_pipeline import prepare_yolo_data, predict_split
    from training.detection_geometry import prepare_image, transform_boxes

    frame = np.zeros((32, 96, 3), dtype=np.uint8)
    frame[:, :, 0] = 230
    frame[:, :, 2] = 15
    cv2.imwrite(str(tmp_path / "source.png"), frame)
    boxes = [[0.1, 0.2, 0.9, 0.8]]
    sample = {"image": "source.png", "split": "validation", "boxes": boxes, "labels": [0]}
    manifest = {"classes": ["firearm"], "samples": [sample]}
    directory = tmp_path / "prepared"
    prepare_yolo_data(tmp_path / "manifest.json", manifest, directory, 96, "letterbox")
    pixels = cv2.imread(str(directory / "images/validation/000000.png"))
    expected, geometry = prepare_image(frame, 96, "letterbox")
    assert np.array_equal(pixels, cv2.cvtColor(expected, cv2.COLOR_RGB2BGR))
    encoded = transform_boxes(boxes, geometry)

    class Fake:
        def predict(self, image, **kwargs):
            assert np.array_equal(image, pixels)
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxyn=torch.tensor(encoded), conf=torch.tensor([0.9]), cls=torch.tensor([0])
                    ),
                    speed={},
                )
            ]

    predictions, targets, _ = predict_split(
        Fake(), tmp_path / "manifest.json", manifest, "validation", 96, "cpu", "letterbox"
    )
    assert torch.allclose(predictions[0]["boxes"], targets[0]["boxes"], atol=1e-6)
    # Validation augmentation must remain disabled even with robust training transforms available.
    from training.detection_pipeline import DetectionImages

    dataset = DetectionImages(tmp_path, [sample], 96, False, "letterbox", "robust-v2")
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_detection_iou_metrics_do_not_mislabel_ap50():
    from training.detection_pipeline import evaluate_detections

    targets = [{"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([0])}]
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.8, 1.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        }
    ]
    low = evaluate_detections(preds, targets, 1)
    high = evaluate_detections(preds, targets, 1, iou_threshold=0.9)
    assert low["per_class"][0]["false_negatives"] == 0 and high["per_class"][0]["false_negatives"] == 1
    assert high["map50"] is None and high["map_iou"] == 0


def test_runtime_routing_keeps_track_ids_through_approved_fallback(client, monkeypatch):
    from test_runtime import make_capture
    import surveilx.model_selection as selection

    runtime, capture = make_capture(client)
    records = {}
    for version, score in (("a", 0.9), ("b", 0.6)):
        record = candidate(version, score, score)
        record["manifest"].update(domain=capture.environment, synthetic=capture.synthetic)
        expert = SimpleNamespace(
            version=version,
            name=version,
            manifest=record["manifest"],
            calibrated=True,
            infer=lambda *args: [Detection([0.1, 0.1, 0.3, 0.6], "person", 0.9)],
        )
        records[version] = record
        runtime.install_model("detection:" + version, expert, version)
    monkeypatch.setattr(
        selection, "evaluation_for", lambda data, version, manifest: records[version]["evaluation"]
    )
    runtime.infer(capture)
    first = capture.outputs[0]["track_id"]
    tracker = capture.tracker

    def fail(*args):
        raise RuntimeError("simulated accelerator error")

    runtime.experts["detection:a"].infer = fail
    capture.sequence += 1
    runtime.infer(capture)
    assert capture.outputs[0]["track_id"] == first and capture.tracker is tracker
    assert capture.calibration == "component_calibrated; pooled_reliability_unvalidated"
    assert capture.category_routes["models"]["a"]["state"] == "failed"
    assert runtime.active_models["detection:a"] == "a"  # failure does not alter deployment approval
    runtime.experts["detection:b"].infer = fail
    capture.sequence += 1
    runtime.infer(capture)
    assert capture.detector == runtime.synthetic_detector.name
    assert capture.category_routes == {} and capture.calibration == "uncalibrated"
    # Cooldown must keep serving the explicit baseline rather than returning no detections.
    capture.sequence += 1
    runtime.infer(capture)
    assert capture.outputs and capture.expert_key[0] == "fallback"


def test_event_fallback_cooldown_does_not_postpone_recovery(client, monkeypatch):
    import time
    from test_runtime import make_capture

    runtime, capture = make_capture(client)
    clock = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    calls = []
    manifest = {"domain": capture.environment, "synthetic": True, "task": "event"}

    def primary(*args, **kwargs):
        calls.append("primary")
        raise RuntimeError("simulated error")

    for version, infer in (("a", primary), ("b", lambda *a, **k: {"decision": "fallback"})):
        runtime.install_model("event:" + version, SimpleNamespace(
            version=version, manifest=manifest, scene=True, infer=infer), version)
    capture.expert_cursor = 0
    runtime.infer(capture)
    assert capture.secondary_result["decision"] == "fallback"
    deadline = runtime.event_failed_until["a"]
    clock[0] += 2
    capture.expert_cursor = 0
    capture.sequence += 1
    runtime.infer(capture)
    assert calls == ["primary"] and runtime.event_failed_until["a"] == deadline
    clock[0] = deadline + 1
    capture.expert_cursor = 0
    capture.sequence += 1
    runtime.infer(capture)
    assert calls == ["primary", "primary"]


def test_condition_probe_sampling_covers_classes_and_empty_images():
    from scripts.evaluate_conditions import stratified_samples

    samples = [{"image": str(i), "labels": [i // 5]} for i in range(20)]
    samples.append({"image": "empty", "labels": []})
    chosen = stratified_samples(samples, 8)
    assert len(chosen) == len({s["image"] for s in chosen}) == 8
    assert {label for s in chosen for label in s["labels"]} == {0, 1, 2, 3}
    assert any(not s["labels"] for s in chosen)
    assert chosen == stratified_samples(samples, 8)
    with pytest.raises(ValueError):
        stratified_samples(samples, 0)


def test_catalog_reads_legacy_yolo_curve_without_inventing_loss(tmp_path):
    import json
    from surveilx.model_selection import catalog
    from training.datasets import digest

    run = tmp_path / "runs" / "legacy"
    (run / "fit").mkdir(parents=True)
    (run / "weights.pt").write_bytes(b"fixture")
    (run / "manifest.json").write_text(json.dumps({"task": "detection", "classes": ["person"],
        "weights_sha256": digest(run / "weights.pt"), "calibrated": False}))
    (run / "fit" / "results.csv").write_text("epoch,metrics/mAP50(B),train/cls_loss\n1,0.2,1.3\n2,0.3,1.1\n")
    report = catalog(tmp_path)
    assert not report["errors"]
    curve = report["models"][0]["learning_curve"]
    assert [r["validation_map50"] for r in curve] == [0.2, 0.3]
    assert curve[-1]["training"]["losses"]["classification"] == 1.1
    assert curve[-1].get("train_loss") is None


def test_retry_queue_waits_for_commit_headroom(monkeypatch):
    import scripts.queue_accuracy_retry as queue

    monkeypatch.setattr(queue, "available_commit_bytes", lambda: 100)
    assert "commit" in queue.resource_wait_reason({"minimum_available_commit_bytes": 200})
    assert queue.resource_wait_reason({"minimum_available_commit_bytes": 50}) is None
