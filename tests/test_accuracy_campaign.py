import json
from types import SimpleNamespace

import pytest

from scripts import train_accuracy_campaign as runner


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path)
    dataset = tmp_path / "datasets/fixture/manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}")
    parent = tmp_path / "runs/parent"
    parent.mkdir(parents=True)
    (parent / "weights.pt").write_bytes(b"parent checkpoint")
    metadata = {"dataset_sha256": runner.digest(dataset), "weights_sha256": runner.digest(parent / "weights.pt"),
                "classes": ["fire", "smoke"], "domain": "safety",
                "metrics": {"map50": 0.2, "per_class": [{"class_id": 0, "ap50": 0.4},
                                                      {"class_id": 1, "ap50": 0.0}]}}
    (parent / "manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(runner, "validate_manifest", lambda path: (
        {"task": "detection", "classes": metadata["classes"], "domain": "safety", "samples": []}, {}))
    monkeypatch.setattr(runner, "register", lambda version: "id-" + version)
    job = {"version": "candidate", "dataset": "fixture", "parent": "parent", "pipeline": "scratch",
           "options": {"epochs": 3, "balanced-sampling": True}, "dataset_sha256": runner.digest(dataset),
           "parent_sha256": metadata["weights_sha256"]}
    directory = tmp_path / "generations/campaign"
    directory.mkdir(parents=True)
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps({"name": "campaign", "jobs": [job]}))
    return plan_path, job, metadata


def test_campaign_reuses_completed_but_rejects_changed_weights(tmp_path, monkeypatch):
    plan, job, metadata = fixture(tmp_path, monkeypatch)
    calls = []

    def complete(command, **kwargs):
        calls.append(command)
        output = tmp_path / "runs/candidate"
        output.mkdir(parents=True)
        (output / "weights.pt").write_bytes(b"candidate checkpoint")
        candidate = {**metadata, "task": "detection", "initialization": job["parent_sha256"],
                     "weights_sha256": runner.digest(output / "weights.pt")}
        candidate["metrics"] = {"map50": 0.3, "per_class": [{"class_id": 0, "ap50": 0.3},
                                                          {"class_id": 1, "ap50": 0.3}]}
        (output / "manifest.json").write_text(json.dumps(candidate))
        return SimpleNamespace(pid=123, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", complete)
    first = runner.run(plan)
    assert first["state"] == "completed"
    result = first["jobs"]["candidate"]["comparison"]
    assert result["delta"] == pytest.approx(0.1)
    assert result["classes"][0]["delta"] == pytest.approx(-0.1)  # regression remains visible
    assert "--balanced-sampling" in calls[0]
    assert runner.run(plan)["state"] == "completed" and len(calls) == 1
    (tmp_path / "runs/candidate/weights.pt").write_bytes(b"corrupted")
    assert runner.run(plan)["state"] == "completed_with_failures"
    assert len(calls) == 1


def test_campaign_preserves_partial_outputs_and_foreign_lock(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    lock = plan.parent / "runner.lock"
    lock.write_text("live-runner")
    with pytest.raises(FileExistsError):
        runner.run(plan)
    assert lock.read_text() == "live-runner"
    lock.unlink()
    output = tmp_path / "runs/candidate"
    output.mkdir(parents=True)
    (output / "weights.pt").write_bytes(b"partial")
    result = runner.run(plan)
    assert result["state"] == "completed_with_failures"
    assert "Partial run preserved" in result["jobs"]["candidate"]["error"]
    assert (output / "weights.pt").read_bytes() == b"partial"


def test_campaign_rejects_plan_mutation_without_rewriting_journal(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    journal = plan.parent / "status.json"
    original = json.dumps({"plan_sha256": "other", "state": "completed"})
    journal.write_text(original)
    with pytest.raises(ValueError, match="plan changed"):
        runner.run(plan)
    assert journal.read_text() == original
