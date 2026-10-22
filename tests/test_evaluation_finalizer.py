import json
from types import SimpleNamespace

import pytest

from scripts import finalize_model_evaluations as worker


def test_evaluation_worker_only_evaluates_after_dependencies(tmp_path, monkeypatch):
    journal = tmp_path / "campaign.json"
    journal.write_text('{"state":"running"}')
    output = tmp_path / "evaluations/finalizer.json"
    monkeypatch.setattr(worker.psutil, "process_iter", lambda *args: [])
    monkeypatch.setattr(worker.psutil, "virtual_memory", lambda: SimpleNamespace(available=5_000_000_000))
    waits = []

    def complete(seconds):
        waits.append(seconds)
        journal.write_text('{"state":"completed_with_failures"}')

    def evaluate(command, **kwargs):
        assert waits == [30]
        assert command[2:] == ["scripts.evaluate_model_catalog"]
        (output.parent / "status.json").write_text('[{"state":"completed"}]')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.time, "sleep", complete)
    monkeypatch.setattr(worker.subprocess, "run", evaluate)
    worker.finalize([journal], output)
    assert json.loads(output.read_text())["state"] == "completed"
    assert not output.with_suffix(".lock").exists()


def test_evaluation_worker_obeys_stop_and_releases_lock(tmp_path, monkeypatch):
    journal = tmp_path / "campaign.json"
    journal.write_text('{"state":"stopped_by_user"}')
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    output = tmp_path / "finalizer.json"
    with pytest.raises(RuntimeError, match="stopped or failed"):
        worker.finalize([journal], output)
    assert not output.with_suffix(".lock").exists()
