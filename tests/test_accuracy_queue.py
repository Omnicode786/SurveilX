import json
from types import SimpleNamespace

import pytest

from scripts import queue_ppe_accuracy as queue


def fixture(tmp_path, monkeypatch, predecessor_state):
    monkeypatch.setattr(queue.settings, "data_dir", tmp_path)
    dataset = tmp_path / "datasets/sh17-development-v2"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}")
    (dataset / "acquisition.json").write_text('{"state":"completed"}')
    previous = tmp_path / "generations/accuracy-g3b-weapons-scratch"
    previous.mkdir(parents=True)
    (previous / "status.json").write_text(json.dumps({"state": predecessor_state, "pid": 123}))
    monkeypatch.setattr(queue, "build_plan", lambda directory, dataset: {"name": directory.name, "jobs": []})
    return tmp_path / "generations/ppe"


def test_queue_starts_after_both_dependencies_and_preserves_plan(tmp_path, monkeypatch):
    directory = fixture(tmp_path, monkeypatch, "completed")
    calls = []
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    result = queue.queue(directory)
    assert result["state"] == "completed"
    assert len(calls) == 1 and "scripts.train_accuracy_campaign" in calls[0]
    assert (directory / "plan.json").exists()
    assert not (directory / "queue.lock").exists()


def test_queue_does_not_overlap_a_live_training_campaign(tmp_path, monkeypatch):
    directory = fixture(tmp_path, monkeypatch, "running")
    monkeypatch.setattr(queue.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: ["python", "-m", "scripts.train_accuracy_campaign"]))
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("Training must not overlap"))

    def stop_waiting(seconds):
        assert seconds == 30
        raise RuntimeError("test stopped after dependency check")

    monkeypatch.setattr(queue.time, "sleep", stop_waiting)
    with pytest.raises(RuntimeError, match="dependency check"):
        queue.queue(directory)
    assert not (directory / "plan.json").exists()
