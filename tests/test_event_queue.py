import json
from types import SimpleNamespace

import pytest

from scripts import queue_event_generation as runner


def setup(tmp_path, monkeypatch, predecessor):
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path)
    dataset = tmp_path / "datasets/clips/manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}")
    monkeypatch.setattr(runner, "validate_manifest", lambda path: ({}, {}))
    directory = tmp_path / "generations/event-g1"
    directory.mkdir(parents=True)
    after = tmp_path / "previous.json"
    after.write_text(json.dumps({"state": predecessor}))
    plan = {
        "generation": directory.name,
        "dataset": "clips",
        "dataset_sha256": runner.digest(dataset),
        "after": str(after),
        "epochs": 20,
        "threads": 2,
        "patience": 5,
        "augment": True,
    }
    (directory / "event-plan.json").write_text(json.dumps(plan))
    return directory, dataset


def test_event_queue_obeys_dependency_and_passes_frozen_recipe(tmp_path, monkeypatch):
    directory, _ = setup(tmp_path, monkeypatch, "completed")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        (directory / "status.json").write_text('{"state":"completed"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.queue(directory)["state"] == "completed"
    assert "--augment" in calls[0] and calls[0][calls[0].index("--threads") + 1] == "2"
    assert not (directory / "queue.lock").exists()


def test_event_queue_refuses_overlap_and_changed_dataset(tmp_path, monkeypatch):
    directory, dataset = setup(tmp_path, monkeypatch, "optimizing")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("must wait"))

    def stop(seconds):
        raise RuntimeError("waiting as required")

    monkeypatch.setattr(runner.time, "sleep", stop)
    with pytest.raises(RuntimeError, match="waiting as required"):
        runner.queue(directory)
    assert not (directory / "queue.lock").exists()
    dataset.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="dataset changed"):
        runner.queue(directory)
