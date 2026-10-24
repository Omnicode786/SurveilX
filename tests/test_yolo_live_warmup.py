from types import SimpleNamespace

import numpy as np
import pytest


def test_live_baseline_warms_before_camera_frames(tmp_path, monkeypatch):
    from surveilx import accelerators
    from surveilx.vision import YOLODetector
    import ultralytics

    checkpoint = tmp_path / "reviewed.pt"
    checkpoint.write_bytes(b"test checkpoint")
    calls = []

    def predict(frame, **kwargs):
        calls.append((frame.copy(), kwargs))
        return []

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: SimpleNamespace(predict=predict))
    monkeypatch.setattr(accelerators, "torch_device", lambda: "cpu")
    detector = YOLODetector(checkpoint)
    assert len(calls) == 1
    frame, args = calls[0]
    assert frame.shape == (320, 320, 3) and frame.dtype == np.uint8 and not frame.any()
    assert args == {"imgsz": 320, "device": "cpu", "verbose": False}
    assert detector.warmup_ms >= 0 and detector.calibrated is False


def test_warmup_failure_propagates_for_explicit_runtime_fallback(tmp_path, monkeypatch):
    from surveilx import accelerators
    from surveilx.vision import YOLODetector
    import ultralytics

    checkpoint = tmp_path / "reviewed.pt"
    checkpoint.write_bytes(b"test checkpoint")

    def fail(*args, **kwargs):
        raise RuntimeError("GPU warmup failed")

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: SimpleNamespace(predict=fail))
    monkeypatch.setattr(accelerators, "torch_device", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU warmup failed"):
        YOLODetector(checkpoint)
