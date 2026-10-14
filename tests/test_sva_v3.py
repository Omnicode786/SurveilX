import copy
import pytest
import torch
from training.detector_model import (
    SVADetector,
    detection_loss,
    flatten_predictions,
    assign_targets,
    decode_detections,
)
from training.detector_v3 import quality_focal_loss


def target(boxes, labels):
    return {"boxes": torch.tensor(boxes).reshape(-1, 4).float(), "labels": torch.tensor(labels).long()}


def test_v3_four_scales_context_and_legacy_round_trip():
    torch.set_num_threads(2)
    for version in (1, 2, 3):
        model = SVADetector(4, width=8, depth=1, architecture_version=version).eval()
        x = torch.rand(1, 3, 64, 96)
        heads = model(x)
        assert len(heads) == (4 if version == 3 else 3)
        clone = SVADetector(**model.config).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(heads[0]["classes"], clone(x)[0]["classes"])
        assert all(torch.isfinite(h["classes"]).all() for h in heads)
        if version == 3:
            assert [h["classes"].shape[-2:] for h in heads] == [(16, 24), (8, 12), (4, 6), (2, 3)]
            with pytest.raises(ValueError):
                model.configure_trainable_scope("classification")
            with pytest.raises(ValueError):
                model(x, torch.zeros(1, 3))


def test_dense_quality_has_background_and_true_class_gradients():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    quality = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
    loss = quality_focal_loss(logits, quality, torch.tensor(1))
    loss.backward()
    assert logits.grad[0, 0] < 0  # increase the matched correct class
    assert logits.grad[0, 1] > 0 and (logits.grad[1] > 0).all()  # suppress wrong/background
    assert torch.isfinite(loss)


def test_v3_synthetic_backward_empty_tiny_overlapping_and_level_coverage():
    model = SVADetector(3, width=8, depth=1, architecture_version=3)
    heads = model(torch.rand(2, 3, 64, 64))
    truths = [
        target(
            [
                [0.001, 0.001, 0.003, 0.003],
                [0.49, 0.49, 0.51, 0.51],
                [0.49, 0.49, 0.51, 0.51],
                [0.05, 0.05, 0.95, 0.95],
            ],
            [0, 1, 2, 0],
        ),
        target([], []),
    ]
    flat = flatten_predictions(heads)
    obj, _, _, owners = assign_targets(flat, truths, True, "coverage-v3")
    assert set(owners[0][owners[0] >= 0].tolist()) == {0, 1, 2, 3}
    assert not obj[1].any()
    assert (owners[0][flat["levels"] == 3] >= 0).any()
    result = detection_loss(heads, truths, quality_warmup=1.0)
    assert result["unassigned_objects"] == 0 and len(result["positives_per_level"]) == 4
    result["loss"].backward()  # no optimizer or real data
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.v3.classifier.weight.grad.abs().sum() > 0
    empty = detection_loss(model(torch.rand(1, 3, 64, 64)), [target([], [])])
    assert empty["classification"] > 0 and torch.isfinite(empty["loss"])
    with pytest.raises(ValueError):
        detection_loss(heads, truths, classification_focal_gamma=2)


def test_quality_decoder_does_not_multiply_auxiliary_objectness():
    model = SVADetector(1, width=8, depth=1, architecture_version=3).eval()
    with torch.no_grad():
        model.v3.classifier.weight.zero_()
        model.v3.classifier.bias.fill_(2)
        model.v3.objectness.weight.zero_()
        model.v3.objectness.bias.fill_(-30)
    heads = model(torch.rand(1, 3, 64, 64))
    decoded = decode_detections(heads, score_threshold=0.5)[0]
    assert len(decoded["scores"]) > 0 and torch.allclose(
        decoded["scores"], torch.full_like(decoded["scores"], torch.sigmoid(torch.tensor(2.0)))
    )
    for head in heads:
        head.pop("quality_scores")
    assert len(decode_detections(heads, score_threshold=0.5)[0]["scores"]) == 0


def test_feature_transfer_preserves_sources_and_resets_heads():
    for version in (1, 2):
        legacy = SVADetector(3, width=8, depth=1, architecture_version=version)
        state = copy.deepcopy(legacy.state_dict())
        new = SVADetector(3, width=8, depth=1, architecture_version=3)
        initial_head = new.v3.classifier.weight.detach().clone()
        keys = new.load_legacy_backbone(state)
        assert keys and torch.equal(new.v3.stem[0][0].weight, legacy.stem[0][0].weight)
        assert torch.equal(new.v3.laterals[1][0].weight, legacy.laterals[0][0].weight)
        assert torch.equal(new.v3.classifier.weight, initial_head)
        assert all(torch.equal(state[k], legacy.state_dict()[k]) for k in state)
        with pytest.raises(RuntimeError):
            new.load_legacy_backbone({k: v for k, v in state.items() if k != "classifier.weight"})


def test_v3_saved_runtime_matches_reference_decoder_and_blocks_legacy_onnx(tmp_path, monkeypatch):
    import json
    import numpy as np
    from training.detection_pipeline import digest
    from training.detection_geometry import prepare_image, transform_boxes
    from surveilx.detector_expert import ScratchDetector
    from surveilx.onnx_detector import ONNXScratchDetector

    monkeypatch.setattr("surveilx.accelerators.torch_device", lambda: "cpu")
    model = SVADetector(1, width=8, depth=1, architecture_version=3).eval()
    with torch.no_grad():
        model.v3.classifier.weight.zero_()
        model.v3.classifier.bias.fill_(2)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    frame = np.zeros((32, 96, 3), dtype=np.uint8)
    for mode in ("stretch", "letterbox"):
        metadata = {
            "task": "detection",
            "architecture": "sva-detector",
            "model_config": model.config,
            "weights_sha256": digest(tmp_path / "weights.pt"),
            "config": {"image_size": 64, "preprocessing": mode},
            "classes": ["fixture"],
            "calibration": {"temperature": 1.0, "bias": 0.0, "threshold": 0.5, "status": "fitted"},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(metadata))
        runtime = ScratchDetector(tmp_path)
        detected = runtime.infer(frame)
        image, geometry = prepare_image(frame, 64, mode)
        reference = decode_detections(
            model(torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None] / 255),
            score_threshold=0.001,
        )[0]
        boxes = transform_boxes(reference["boxes"].numpy(), geometry, inverse=True)
        valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
        np.testing.assert_allclose([d.box for d in detected], boxes[valid], atol=1e-6)
        np.testing.assert_allclose(
            [d.score for d in detected], reference["scores"].numpy()[valid], atol=1e-6
        )
        with pytest.raises(ValueError, match="v3 four-level"):
            ONNXScratchDetector(tmp_path, tmp_path / "missing-export", ["CPUExecutionProvider"])
