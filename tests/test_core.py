import numpy as np
import pytest

from surveilx.controller import Candidate, Scheduler
from surveilx.hardware import Hardware, PowerGovernor
from surveilx.vision import Detection, SyntheticDetector, Tracker, generated_frame
from training.calibration import arbitrate, fit_temperature, metrics
from training.datasets import generate, validate_manifest


def hardware(**changes):
    values = dict(
        os="Windows",
        architecture="AMD64",
        cores=8,
        ram_gb=16,
        available_gb=8,
        cpu_percent=20,
        memory_percent=40,
        gpu=None,
        temperature_c=None,
        battery_percent=None,
        plugged_in=None,
    )
    return Hardware(**(values | changes))


def test_power_hysteresis_and_throttling():
    governor = PowerGovernor()
    for _ in range(20):
        result = governor.update(hardware())
    assert result["level"] == "performance"
    assert governor.update(hardware(temperature_c=90))["level"] == "balanced"
    assert governor.update(hardware(available_gb=0.2))["level"] == "economy"
    assert governor.update(hardware())["level"] == "economy"


def test_low_memory_pc_does_not_promote():
    governor = PowerGovernor()
    for _ in range(100):
        result = governor.update(hardware(ram_gb=2, available_gb=1))
    assert result["level"] == "economy"


def test_scheduler_budget_and_starvation():
    scheduler = Scheduler()
    result = scheduler.allocate([Candidate("high", 5, 1, 90), Candidate("old", 1, 7, 90)], 100, 5)
    assert result["selected"] == ["old"]
    assert result["estimated_ms"] <= 100
    result = scheduler.allocate([Candidate("old", 1, 7, 300)], 100, 5)
    assert result["deferred"][0]["coverage_violation"]


def test_tracker_one_to_one_and_expiry():
    tracker = Tracker()
    old = tracker.update([Detection([0, 0, 0.5, 0.5], "person", 0.9)], now=0)[0].track_id
    detections = tracker.update(
        [Detection([0, 0, 0.5, 0.5], "person", 0.9), Detection([0, 0, 0.5, 0.5], "person", 0.9)], now=1
    )
    assert len({d.track_id for d in detections}) == 2
    assert detections[0].track_id == old
    assert tracker.update([Detection([0, 0, 0.5, 0.5], "person", 0.9)], now=4)[0].track_id != old


def test_synthetic_detector_is_explicit():
    detections = SyntheticDetector().infer(generated_frame(0, 2))
    assert detections and detections[0].label == "synthetic_entity"


def test_calibration_and_abstention():
    logits = np.tile([[2.0, -1.0], [-1.0, 2.0]], (10, 1))
    labels = np.tile([0, 1], 10)
    temperature = fit_temperature(logits, labels)
    assert metrics(logits, labels, temperature)["brier"] <= metrics(logits, labels)["brier"]
    assert arbitrate([{"calibrated": False}])["decision"] == "abstain"
    items = [
        {"calibrated": True, "task": "event", "domain": "test", "classes": ["a", "b"], "probabilities": p}
        for p in [[0.9, 0.1], [0.1, 0.9]]
    ]
    assert arbitrate(items)["decision"] == "review"


def test_manifest_rejects_group_leak(tmp_path):
    import json

    path = generate(tmp_path / "generated", 80)
    manifest, counts = validate_manifest(path)
    assert counts == dict.fromkeys(["train", "validation", "calibration", "test"], 20)
    manifest["samples"][20]["group"] = manifest["samples"][0]["group"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaks"):
        validate_manifest(path)


def test_sva_gradient_reaches_visual_encoder():
    torch = pytest.importorskip("torch")
    from training.models import SVANet

    model = SVANet()
    output = model(torch.rand(2, 4, 3, 32, 32), torch.rand(2, 4, 2, 4), torch.rand(2, 4))
    torch.nn.functional.cross_entropy(output["event"], torch.tensor([0, 1])).backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert output["relations"].shape == (2, 2, 2, 2)


def test_dataset_import_rejects_traversal(tmp_path):
    import io
    import zipfile
    from training.import_bundle import import_bundle

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        import_bundle(buffer.getvalue(), tmp_path / "import")
    assert not (tmp_path / "import").exists()
