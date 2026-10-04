import sys
import json

import pytest

from surveilx import training_runtime as runtime


def test_training_interpreter_and_batch_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.settings, "training_yolo_batch_size", None)
    monkeypatch.setattr(runtime.settings, "training_event_batch_size", None)
    monkeypatch.setattr(runtime.settings, "training_python", "")
    monkeypatch.setattr(runtime.settings, "training_batch_size", None)
    assert runtime.training_interpreter() == sys.executable
    assert runtime.training_batch_arguments() == []
    path = tmp_path / "python.exe"
    monkeypatch.setattr(runtime.settings, "training_python", str(path))
    with pytest.raises(ValueError, match="missing"):
        runtime.training_interpreter()
    assert "missing" in runtime.training_runtime_status()["error"]
    path.write_bytes(b"test fixture; never executed")
    monkeypatch.setattr(runtime.settings, "training_batch_size", 1)
    assert runtime.training_interpreter() == str(path.resolve())
    assert runtime.training_batch_arguments() == ["--batch-size", "1"]
    assert runtime.training_batch_arguments("training.yolo_pipeline") == ["--batch", "1"]
    monkeypatch.setattr(runtime.settings, "training_batch_size", 0)
    with pytest.raises(ValueError, match="batch size"):
        runtime.training_batch_arguments()


def test_event_batch_reduces_for_larger_user_contracts(monkeypatch):
    monkeypatch.setattr(runtime.settings, "training_event_batch_size", 8)
    assert runtime.training_batch_arguments(
        "training.pipeline", {"input_contract": {"frames": 8, "image_size": 96}}
    ) == ["--batch-size", "8"]
    assert runtime.training_batch_arguments(
        "training.pipeline", {"input_contract": {"frames": 64, "image_size": 224}}
    ) == ["--batch-size", "1"]


def test_gpu_evidence_must_match_configured_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.settings, "data_dir", tmp_path)
    path = tmp_path / "python.exe"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(runtime.settings, "training_python", str(path))
    report_path = tmp_path / "hardware/cuda-training-g1/report.json"
    report_path.parent.mkdir(parents=True)
    report = {
        "state": "completed",
        "python": str(tmp_path / "another.exe"),
        "scope": "recorded offline probe",
        "created": 1,
        "torch": "fixture",
        "cuda_runtime": "fixture",
        "gpu": "fixture",
        "precision": "float32",
        "results": [],
    }
    report_path.write_text(json.dumps(report))
    assert runtime.training_runtime_status()["recorded_validation"] is None
    report["python"] = str(path)
    report_path.write_text(json.dumps(report))
    assert runtime.training_runtime_status()["recorded_validation"]["gpu"] == "fixture"
    report["state"] = "failed"
    report_path.write_text(json.dumps(report))
    assert runtime.training_runtime_status()["recorded_validation"] is None
