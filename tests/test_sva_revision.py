import copy

import cv2
import numpy as np
import pytest
import torch

from training.detector_model import SVADetector, assign_targets, detection_loss, flatten_predictions
from training.detection_geometry import prepare_image, prepare_zone, transform_boxes
from training.detection_pipeline import DetectionImages, collate_detection, predict
from training.observability import EpochLossSummary, ValidationSelection
from scripts.watch_training import format_epoch


def test_numpy_historical_audit_matches_tensor_assignment():
    from scripts.analyze_sva_datasets import grid, historical_assignment
    points, steps, levels = grid(64)
    boxes = [[.4, .4, .6, .6], [.45, .45, .55, .55], [.001, .001, .002, .002], [.1, .49, .9, .51]]
    flat = {"points": torch.from_numpy(points), "steps": torch.from_numpy(steps),
            "levels": torch.from_numpy(levels), "classes": torch.zeros(1, len(points), 4),
            "boxes": torch.zeros(1, len(points), 4)}
    target = {"boxes": torch.tensor(boxes), "labels": torch.arange(4)}
    owners = assign_targets(flat, [target], return_metadata=True)[3][0].numpy()
    np.testing.assert_array_equal(owners, historical_assignment(boxes, 64)["point_winners"])


def test_augmentation_flips_pixels_boxes_and_zones_together(tmp_path, monkeypatch):
    frame = np.zeros((32, 96, 3), np.uint8)
    frame[:, :30] = [20, 60, 120]
    zone = np.zeros((32, 96), np.uint8)
    zone[:, :30] = 255
    cv2.imwrite(str(tmp_path / "image.png"), frame)
    cv2.imwrite(str(tmp_path / "zone.png"), zone)
    sample = {"image": "image.png", "zone_map": "zone.png", "boxes": [[.05, .2, .3, .8]], "labels": [0]}
    monkeypatch.setattr("training.detection_pipeline.random.random", lambda: 0.)
    monkeypatch.setattr("training.detection_pipeline.random.uniform", lambda low, high: 1. if low > 0 else 0.)
    for mode in ("stretch", "letterbox"):
        original = DetectionImages(tmp_path, [sample], 64, preprocessing=mode)[0]
        flipped = DetectionImages(tmp_path, [sample], 64, augment=True, preprocessing=mode)[0]
        assert torch.equal(flipped[0], original[0].flip(-1))
        assert torch.equal(flipped[2], original[2].flip(-1))
        assert torch.allclose(flipped[3]["boxes"][:, [0, 2]], 1 - original[3]["boxes"][:, [2, 0]])


def test_failure_categories_and_duplicate_matching():
    from scripts.diagnose_sva_failures import attribute_failures, matched_ground_truth
    target = {"boxes": torch.tensor([[.1, .1, .5, .5]]), "labels": torch.tensor([0])}
    empty = {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    calibration = {"temperature": 1., "bias": 0., "threshold": .5}
    for category, box, label, score, keep in [
        ("unlocalized", [.7, .7, .9, .9], 0, .9, False),
        ("localized_wrong_class", [.1, .1, .5, .5], 1, .9, False),
        ("correct_class_confidence_too_low", [.1, .1, .5, .5], 0, .1, False),
        ("duplicate_nms_or_matching", [.1, .1, .5, .5], 0, .9, False),
        ("correct_detection", [.1, .1, .5, .5], 0, .9, True),
    ]:
        raw = {"boxes": torch.tensor([box]), "scores": torch.tensor([score]), "labels": torch.tensor([label]),
               "objectness": torch.tensor([.9]), "class_probabilities": torch.tensor([[.9, .1]])}
        rows, _ = attribute_failures(raw, raw if keep else empty, target, calibration)
        assert rows[0]["category"] == category
    duplicate = {k: v.repeat(2, 1) if v.ndim == 2 else v.repeat(2) for k, v in raw.items()}
    matched, repeats, other = matched_ground_truth(duplicate, target, .5)
    assert matched == {0} and repeats == 1 and other == 0


def test_v1_upgrade_is_identity_and_only_classification_has_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    old = SVADetector(classes=4, width=8, depth=1).eval()
    source = copy.deepcopy(old.state_dict())
    upgraded = SVADetector(classes=4, width=8, depth=1, architecture_version=2).eval()
    upgraded.load_v1_for_classification_upgrade(source)
    upgraded.configure_trainable_scope("classification")
    images = torch.rand(1, 3, 64, 64)
    expected, observed = old(images), upgraded(images)
    for a, b in zip(expected, observed, strict=True):
        for key in a:
            assert torch.equal(a[key], b[key]), key
    loss = detection_loss(observed, [{"boxes": torch.tensor([[.2, .2, .8, .8]]), "labels": torch.tensor([2])}])
    loss["loss"].backward()  # Tiny synthetic correctness check, no optimizer step.
    for name, parameter in upgraded.named_parameters():
        if name.startswith(("class_tower.", "classifier.", "class_refinement.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert upgraded.class_refinement.project.weight.grad.abs().sum() > 0
    for key in source:
        assert torch.equal(old.state_dict()[key], source[key])
    assert "architecture_version" not in old.config
    clone = SVADetector(**old.config)
    clone.load_state_dict(source, strict=True)
    with pytest.raises(ValueError, match="exact v1"):
        upgraded.load_v1_for_classification_upgrade({})


def test_coverage_repairs_nested_objects_without_changing_eligible_locations():
    model = SVADetector(classes=2, width=8, depth=1)
    heads = model(torch.zeros(1, 3, 64, 64))
    flat = flatten_predictions(heads)
    targets = [{"boxes": torch.tensor([[.4, .4, .6, .6], [.45, .45, .55, .55]]), "labels": torch.tensor([0, 1])}]
    original = assign_targets(flat, targets, return_metadata=True)
    repaired = assign_targets(flat, targets, return_metadata=True, assignment="coverage-v2")
    assert set(original[3][original[3] >= 0].tolist()) == {1}
    assert set(repaired[3][repaired[3] >= 0].tolist()) == {0, 1}
    assert torch.equal(original[0], repaired[0])
    measured = detection_loss(heads, targets, assignment="coverage-v2")
    assert measured["assigned_objects"] == 2 and measured["unassigned_objects"] == 0
    assert measured["positives_per_level"].sum() == measured["positives"]


def test_assignment_handles_tiny_elongated_overcrowded_and_empty_targets():
    model = SVADetector(classes=2, width=8, depth=1)
    heads = model(torch.zeros(1, 3, 64, 64))
    for boxes in ([[.501, .501, .502, .502]], [[.1, .49, .9, .51]], [[.49, .49, .51, .51]] * 30, []):
        targets = [{"boxes": torch.tensor(boxes).reshape(-1, 4), "labels": torch.zeros(len(boxes), dtype=torch.long)}]
        result = detection_loss(heads, targets, assignment="coverage-v2")
        assert torch.isfinite(result["loss"])
        assert result["assigned_objects"] + result["unassigned_objects"] == len(boxes)
        if not boxes:
            assert result["positives"] == 0
        elif len(boxes) == 1:
            assert result["unassigned_objects"] == 0
        else:
            assert result["unassigned_objects"] > 0  # Capacity limits remain explicit.


def test_geometry_preserves_legacy_pixels_and_letterbox_roundtrip():
    image = np.random.default_rng(1).integers(0, 256, (31, 97, 3), dtype=np.uint8)
    zones = np.arange(31 * 97, dtype=np.float32).reshape(31, 97)
    boxes = np.array([[.1, .2, .8, .9]], dtype=np.float32)
    legacy, g = prepare_image(image, 64)
    assert np.array_equal(legacy, cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2RGB))
    assert np.array_equal(prepare_zone(zones, g), cv2.resize(zones, (64, 64)))
    assert np.array_equal(transform_boxes(boxes, g), boxes)
    padded, geometry = prepare_image(image, 64, "letterbox")
    assert padded.shape == (64, 64, 3)
    assert np.allclose(transform_boxes(transform_boxes(boxes, geometry), geometry, inverse=True), boxes, atol=1e-6)
    assert np.all(prepare_zone(zones, geometry)[:geometry["top"]] == 0)


def test_letterbox_dataset_runtime_pixels_and_prediction_coordinates_match(tmp_path):
    from surveilx.detector_expert import ScratchDetector
    from surveilx import detector_expert
    from training import detection_pipeline

    frame = np.full((32, 96, 3), 127, np.uint8)
    cv2.imwrite(str(tmp_path / "fixture.png"), frame)
    sample = {"image": "fixture.png", "boxes": [[.25, .25, .75, .75]], "labels": [0]}
    dataset = DetectionImages(tmp_path, [sample], 64, preprocessing="letterbox")
    tensor, _, _, truth = dataset[0]
    class FixedModel(torch.nn.Module):
        def forward(self, images, *args):
            assert torch.equal(images[0], tensor)
            return images
    model = FixedModel()
    prediction = {"boxes": truth["boxes"], "scores": torch.tensor([.9]), "labels": torch.tensor([0]), "embeddings": torch.zeros(1, 1)}
    from unittest.mock import patch
    with patch("training.detector_model.decode_detections", return_value=[prediction]), patch.object(detection_pipeline, "decode_detections", return_value=[prediction]):
        expert = object.__new__(ScratchDetector)
        expert.model, expert.device, expert.classes = model, "cpu", ["fixture"]
        expert.manifest = {"config": {"image_size": 64, "preprocessing": "letterbox"}}
        expert.calibration = {"threshold": .1}
        result = detector_expert.ScratchDetector.infer(expert, frame)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_detection)
        predictions, targets, _ = predict(model, loader, "cpu")
    assert np.allclose(result[0].box, sample["boxes"][0], atol=1e-6)
    assert np.allclose(predictions[0]["boxes"], targets[0]["boxes"], atol=1e-6)


def test_patience_parent_fallback_and_epoch_twelve_stop_at_twenty():
    fallback = ValidationSelection(8, .08, 4)
    for epoch in range(1, 9):
        assert not fallback.observe(epoch, .07, 1)
        assert fallback.should_stop is (epoch == 8)
    assert fallback.selected == "parent" and fallback.best_epoch == 0
    fire = ValidationSelection(8, .2, 4)
    for epoch in range(1, 21):
        fire.observe(epoch, .2 + min(epoch, 12) / 100, 3 if epoch <= 12 else 4)
        assert fire.should_stop is (epoch == 20)
    assert fire.best_epoch == 12
    assert fire.report(20, 30)["stopping_reason"] == "validation_patience_exhausted"
    with pytest.raises(ValueError, match="finite"):
        fire.observe(21, float("nan"), 1)


def test_loss_summary_and_legacy_history_format_are_honest():
    summary = EpochLossSummary()
    summary.update({"loss": torch.tensor(2.), "positives": 0, "positives_per_level": [0, 0, 0]}, 1)
    assert summary.report()["positive_assignments"] == 0
    assert summary.report()["losses"]["classification"] is None
    lines = format_epoch({"epoch": 8, "train_loss": 2, "validation_map50": .05, "elapsed_seconds": 10})
    assert "not recorded" in "\n".join(lines)
    assert "loss=2" in "\n".join(lines)
