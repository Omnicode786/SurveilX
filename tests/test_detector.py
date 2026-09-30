import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.detection_pipeline import (  # noqa: E402
    evaluate_detections,
    fit_detection_calibration,
    validate_detection_manifest,
)
from training.detector_model import (  # noqa: E402
    SVADetector,
    box_iou,
    decode_detections,
    detection_loss,
    generalized_iou_loss,
    nms,
)


def test_scratch_detector_has_multiscale_supervision_and_context_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = SVADetector(classes=2, width=8, depth=1)
    output = model(torch.rand(2, 3, 96, 96), torch.ones(2, 4), torch.ones(2, 1, 96, 96))
    assert [o["box_raw"].shape[-1] for o in output] == [12, 6, 3]
    targets = [{"boxes": torch.tensor([[0.1, 0.15, 0.7, 0.8]]), "labels": torch.tensor([1])}] * 2
    losses = detection_loss(output, targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for layer in (
        model.stem[0][0],
        model.regressor,
        model.classifier,
        model.objectness,
        model.conditioners[2].scene,
        model.conditioners[2].zone,
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    predictions = decode_detections(output, score_threshold=0, max_detections=10, pre_nms_topk=30)
    assert len(predictions) == 2
    for prediction in predictions:
        assert len(prediction["boxes"]) <= 10
        assert torch.all(prediction["boxes"] >= 0) and torch.all(prediction["boxes"] <= 1)
        assert prediction["embeddings"].shape[1] == 32


def test_detection_loss_supports_empty_and_tiny_targets():
    model = SVADetector(width=8, depth=1)
    output = model(torch.rand(2, 3, 64, 64))
    targets = [
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
        {"boxes": torch.tensor([[0.501, 0.502, 0.505, 0.506]]), "labels": torch.tensor([0])},
    ]
    result = detection_loss(output, targets)
    assert result["positives"] >= 1
    result["loss"].backward()
    assert torch.isfinite(result["loss"])


def test_iou_giou_and_nms_known_geometry():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1]])
    assert box_iou(boxes[:1], boxes[1:2]).item() == pytest.approx(1)
    assert generalized_iou_loss(boxes[:1], boxes[1:2]).item() == pytest.approx(0)
    assert nms(boxes, torch.tensor([0.9, 0.8, 0.7])).tolist() == [0, 2]


def test_ap50_counts_duplicate_and_wrong_class_as_false_positives():
    targets = [{"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [0]}]
    predictions = [{"boxes": [[0.0, 0.0, 1.0, 1.0]] * 3, "scores": [0.9, 0.8, 0.7], "labels": [1, 0, 0]}]
    result = evaluate_detections(predictions, targets, 2)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    assert result["map50"] == 1  # Absent class excluded from mean AP, included in FP counts.
    predictions[0]["scores"] = [0.9, 0.8, 0.95]
    predictions[0]["boxes"][2] = [0.0, 0.0, 0.1, 0.1]
    assert evaluate_detections(predictions, targets, 2)["map50"] == pytest.approx(0.5)


def test_no_detections_have_zero_ap_and_all_false_negatives():
    prediction = {"boxes": [], "scores": [], "labels": []}
    target = {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [0]}
    metrics = evaluate_detections([prediction], [target], 1)
    assert metrics["map50"] == 0 and metrics["false_negatives"] == 1
    calibration = fit_detection_calibration([prediction], [target])
    assert calibration["status"].startswith("insufficient")


def test_native_nms_matches_python_fallback():
    from training.detector_model import native_nms, python_nms

    if native_nms is None:
        pytest.skip("Optional TorchVision native kernel unavailable")
    torch.manual_seed(42)
    xy = torch.rand(120, 2) * 0.7
    boxes = torch.cat([xy, xy + torch.rand(120, 2) * 0.3], 1)
    scores = torch.rand(120)
    assert torch.equal(nms(boxes, scores), python_nms(boxes, scores))


def test_manifest_rejects_group_leakage_and_image_content_leakage(tmp_path):
    samples = []
    for index, split in enumerate(("train", "validation", "calibration", "test")):
        image = f"{split}.png"
        cv2.imwrite(str(tmp_path / image), np.full((16, 16, 3), index * 20, dtype=np.uint8))
        samples.append(
            {"image": image, "boxes": [[0.1, 0.1, 0.9, 0.9]], "labels": [0], "split": split, "group": split}
        )
    manifest = {
        "schema_version": 1,
        "task": "detection",
        "classes": ["person"],
        "samples": samples,
        "domain": "test",
        "license": "generated test fixture",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _, counts, hashes = validate_detection_manifest(path)
    assert sum(counts.values()) == len(hashes) == 4
    manifest["samples"][1]["group"] = "train"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="group leaks"):
        validate_detection_manifest(path)
    manifest["samples"][1]["group"] = "validation"
    manifest["samples"][1]["image"] = "train.png"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content leaks"):
        validate_detection_manifest(path)
