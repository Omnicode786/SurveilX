import json

import pytest

from scripts import optimize_model_profiles as optimizer


def test_optimizer_runner_is_durable_and_reuses_completed_jobs(tmp_path, monkeypatch):
    directory = tmp_path / "profiles"
    directory.mkdir()
    plan = directory / "plan.json"
    jobs = [{"version": "one", "source": "parent-one"}, {"version": "two", "source": "parent-two"}]
    plan.write_text(json.dumps({"name": "profiles", "jobs": jobs}))
    calls = []

    def complete(job):
        calls.append(job["version"])
        return "id-" + job["version"], {"inference_profiles": [{"tier": "performance"}]}

    monkeypatch.setattr(optimizer, "optimize", complete)
    result = optimizer.run(plan)
    assert result["state"] == "completed"
    assert calls == ["one", "two"]
    assert not (directory / "runner.lock").exists()


def test_optimizer_preserves_foreign_lock(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    plan = directory / "plan.json"
    plan.write_text(json.dumps({"name": "profiles", "jobs": [{"version": "one"}]}))
    lock = directory / "runner.lock"
    lock.write_text("other")
    with pytest.raises(FileExistsError):
        optimizer.run(plan)
    assert lock.read_text() == "other"


def test_yolo_latency_includes_preprocess_inference_and_postprocess():
    assert optimizer.mean_yolo_latency(
        [{"preprocess": 1, "inference": 3, "postprocess": 2}, {"preprocess": 2, "inference": 4, "postprocess": 2}]
    ) == pytest.approx(7)
