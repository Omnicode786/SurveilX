import json
from types import SimpleNamespace

import pytest

from scripts import queue_accuracy_retry as queue


def setup_queue(tmp_path, state):
    directory = tmp_path / "generations/retry"
    directory.mkdir(parents=True)
    (directory / "plan.json").write_text('{"jobs":[{"version":"retry"}]}')
    predecessor = tmp_path / "generations/original/status.json"
    predecessor.parent.mkdir(parents=True)
    predecessor.write_text(json.dumps({"state": state}))
    return directory, predecessor


def test_retry_runs_only_after_terminal_predecessor(tmp_path, monkeypatch):
    directory, predecessor = setup_queue(tmp_path, "completed_with_failures")
    (directory / "status.json").write_text('{"state":"completed"}')
    calls = []
    monkeypatch.setattr(
        queue.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    result = queue.queue(directory, predecessor)
    assert result["state"] == "completed"
    assert len(calls) == 1 and "scripts.train_accuracy_campaign" in calls[0]
    assert not (directory / "queue.lock").exists()


def test_retry_does_not_overlap_running_predecessor(tmp_path, monkeypatch):
    directory, predecessor = setup_queue(tmp_path, "running")
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not overlap"))

    def stop_waiting(seconds):
        assert seconds == 30
        raise RuntimeError("test stopped after dependency check")

    monkeypatch.setattr(queue.time, "sleep", stop_waiting)
    with pytest.raises(RuntimeError, match="dependency check"):
        queue.queue(directory, predecessor)
    assert not (directory / "queue.lock").exists()


def test_retry_rejects_plan_mutation_while_waiting(tmp_path, monkeypatch):
    directory, predecessor = setup_queue(tmp_path, "running")
    monkeypatch.setattr(
        queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("Changed plan must not launch")
    )

    def complete_with_changed_plan(seconds):
        (directory / "plan.json").write_text('{"jobs":[{"version":"changed"}]}')
        predecessor.write_text('{"state":"completed"}')

    monkeypatch.setattr(queue.time, "sleep", complete_with_changed_plan)
    with pytest.raises(ValueError, match="plan changed"):
        queue.queue(directory, predecessor)
    assert json.loads((directory / "queue.json").read_text())["state"] == "failed"
    assert not (directory / "queue.lock").exists()
