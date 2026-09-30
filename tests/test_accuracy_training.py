import copy

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.calibration import metrics  # noqa: E402
from training.detection_pipeline import class_balanced_weights  # noqa: E402
from training.pipeline import Clips  # noqa: E402


def test_rare_class_sampling_is_bounded_and_excludes_holdouts():
    samples = [{"split": "train", "labels": [0]} for _ in range(100)]
    samples += [{"split": "train", "labels": [1, 1]}, {"split": "train", "labels": []}]
    weights = class_balanced_weights(samples, 3)
    assert weights[:100] == [1.0] * 100
    assert weights[-2:] == [4.0, 1.0]
    with pytest.raises(ValueError, match="training split"):
        class_balanced_weights(samples + [{"split": "test", "labels": [1]}], 3)


def test_video_augmentation_preserves_motion_and_entity_boxes(tmp_path):
    frame = np.full((3, 8, 8), 0.4, dtype=np.float32)
    clip = np.stack([frame] * 8)
    boxes = np.tile([[[0.1, 0.2, 0.6, 0.7]]], (8, 1, 1)).astype(np.float32)
    np.savez(tmp_path / "clip.npz", clip=clip, boxes=boxes, context=np.zeros(4))
    dataset = Clips(tmp_path, [{"file": "clip.npz", "label": 1}], augment=True)
    pixels, observed_boxes, _, label = dataset[0]
    assert torch.equal(pixels[0], pixels[-1])
    assert torch.equal(observed_boxes, torch.from_numpy(boxes))
    assert label == 1 and 0 <= pixels.min() <= pixels.max() <= 1


def test_event_report_exposes_missed_class_even_when_accuracy_is_high():
    report = metrics(np.array([[4.0, -4.0]] * 10), np.array([0] * 9 + [1]))
    assert report["accuracy"] == 0.9
    assert report["balanced_accuracy"] == 0.5
    assert report["per_class"][1]["recall"] == 0
    assert report["confusion_matrix"] == [[9, 0], [1, 0]]


def test_yolo_preparation_matches_runtime_geometry_without_moving_labels(tmp_path):
    pytest.importorskip("ultralytics")
    from training.yolo_pipeline import prepare_yolo_data

    cv2.imwrite(str(tmp_path / "wide.png"), np.full((24, 96, 3), 128, np.uint8))
    manifest = {"classes": ["firearm"], "samples": [{"image": "wide.png", "split": "train",
                "boxes": [[0.25, 0.25, 0.75, 0.75]], "labels": [0]}]}
    prepare_yolo_data(tmp_path / "manifest.json", manifest, tmp_path / "prepared", image_size=64)
    assert cv2.imread(str(tmp_path / "prepared/images/train/000000.png")).shape == (64, 64, 3)
    assert (tmp_path / "prepared/labels/train/000000.txt").read_text() == "0 0.5 0.5 0.5 0.5"


def test_adapted_yolo_continuation_restores_backbone_and_adapters(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel
    from training.yolo_model import ConditionedBlock, install_adapters
    from training.yolo_pipeline import ConditionedTrainer, DetectionTrainer

    torch.set_num_threads(2)
    original = DetectionModel("yolo11n.yaml", nc=2, verbose=False)
    parent = install_adapters(copy.deepcopy(original))
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.add_(0.123)
    monkeypatch.setattr(DetectionTrainer, "get_model", lambda *args: copy.deepcopy(original))
    restored = ConditionedTrainer.get_model(object.__new__(ConditionedTrainer), weights=parent, verbose=False)
    assert sum(isinstance(module, ConditionedBlock) for module in restored.modules()) == 3
    for name, expected in parent.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected), name


def test_yolo_thread_budget_applies_after_library_device_setup(monkeypatch):
    pytest.importorskip("ultralytics")
    from types import SimpleNamespace
    from training.yolo_pipeline import bound_threads

    callbacks, observed = {}, []
    model = SimpleNamespace(add_callback=lambda name, function: callbacks.update({name: function}))
    monkeypatch.setattr(torch, "set_num_threads", observed.append)
    assert bound_threads(model, 3) is model
    callbacks["on_pretrain_routine_start"](None)
    callbacks["on_predict_start"](None)
    assert observed == [3, 3]
