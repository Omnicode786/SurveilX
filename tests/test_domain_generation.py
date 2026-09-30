import json
from types import SimpleNamespace

import pytest

from scripts import train_domain_generation as runner


def setup_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path)
    dataset = tmp_path / "datasets" / "fixture"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "validate_manifest", lambda path: ({"task": "detection"}, {"train": 1}))
    monkeypatch.setattr(runner, "register", lambda version: "id-" + version)
    return dataset


def test_generation_runs_serially_then_reuses_completed_artifacts(tmp_path, monkeypatch):
    dataset = setup_runner(tmp_path, monkeypatch)
    commands = []

    def complete(command, **kwargs):
        commands.append(command)
        output = (
            tmp_path / "runs" / ("g1-scratch" if "training.detection_pipeline" in command else "g1-yolo-rai")
        )
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({"dataset_sha256": runner.digest(dataset / "manifest.json")})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", complete)
    assert runner.run("fixture", "g1")["state"] == "completed"
    assert len(commands) == 2
    assert "training.detection_pipeline" in commands[0]
    assert "training.yolo_pipeline" in commands[1]
    assert runner.run("fixture", "g1")["state"] == "completed"
    assert len(commands) == 2
    assert not (tmp_path / "generations/g1/runner.lock").exists()


def test_generation_preserves_partial_run_and_reports_failure(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch)
    output = tmp_path / "runs/g1-scratch"
    output.mkdir(parents=True)
    checkpoint = output / "weights.pt"
    checkpoint.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Interrupted run preserved"):
        runner.run("fixture", "g1")
    journal = json.loads((tmp_path / "generations/g1/status.json").read_text())
    assert journal["state"] == "failed"
    assert checkpoint.read_bytes() == b"preserve"
    assert not (tmp_path / "generations/g1/runner.lock").exists()


def test_generation_resumes_only_known_yolo_last_checkpoint(tmp_path, monkeypatch):
    dataset = setup_runner(tmp_path, monkeypatch)
    scratch = tmp_path / "runs/g1-scratch"
    scratch.mkdir(parents=True)
    (scratch / "manifest.json").write_text(
        json.dumps({"dataset_sha256": runner.digest(dataset / "manifest.json")})
    )
    checkpoint = tmp_path / "runs/g1-yolo-rai/fit/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial")
    commands = []

    def complete(command, **kwargs):
        commands.append(command)
        output = tmp_path / "runs/g1-yolo-rai"
        (output / "manifest.json").write_text(
            json.dumps({"dataset_sha256": runner.digest(dataset / "manifest.json")})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", complete)
    result = runner.run("fixture", "g1")
    assert result["state"] == "completed"
    assert len(commands) == 1
    assert commands[0][-2:] == ["--resume-checkpoint", str(checkpoint)]
    assert result["jobs"]["yolo-rai"]["resumed"] is True


def test_generation_lock_prevents_duplicate_training(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch)
    directory = tmp_path / "generations/g1"
    directory.mkdir(parents=True)
    lock = directory / "runner.lock"
    lock.write_text("123")
    with pytest.raises(FileExistsError):
        runner.run("fixture", "g1")
    assert lock.read_text() == "123"


def test_different_configuration_does_not_rewrite_existing_journal(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch)
    directory = tmp_path / "generations/g1"
    directory.mkdir(parents=True)
    path = directory / "status.json"
    original = json.dumps({"state": "completed", "config": {"epochs": 10}})
    path.write_text(original)
    with pytest.raises(ValueError, match="configuration changed"):
        runner.run("fixture", "g1")
    assert path.read_text() == original
