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


def test_campaign_rejects_changed_or_external_source(tmp_path, monkeypatch):
    _, job, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "trainer.py"
    source.write_text("frozen recipe")
    job["code_sha256"] = {"trainer.py": runner.digest(source)}
    runner.verify_job(job)
    source.write_text("changed recipe")
    with pytest.raises(ValueError, match="Training source changed"):
        runner.verify_job(job)
    job["code_sha256"] = {"../outside.py": "irrelevant"}
    with pytest.raises(ValueError, match="Training source changed"):
        runner.verify_job(job)


def test_campaign_reuses_completed_but_rejects_changed_weights(tmp_path, monkeypatch):
    plan, job, metadata = fixture(tmp_path, monkeypatch)
    calls = []

    def complete(command, **kwargs):
        calls.append(command)
        output = tmp_path / "runs/candidate"
        output.mkdir(parents=True)
        (output / "weights.pt").write_bytes(b"candidate checkpoint")
        candidate = {**metadata, "task": "detection", "architecture": "sva-detector",
                     "config": {"epochs": 3, "balanced_sampling": True}, "initialization": job["parent_sha256"],
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


def test_campaign_resumes_only_yolo_from_its_known_last_checkpoint(tmp_path, monkeypatch):
    plan, job, metadata = fixture(tmp_path, monkeypatch)
    job["pipeline"] = "yolo"
    job["options"] = {"epochs": 3, "imgsz": 384, "batch": 8}
    plan.write_text(json.dumps({"name": "campaign", "jobs": [job]}))
    checkpoint = tmp_path / "runs/candidate/fit/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"interrupted checkpoint")
    calls = []

    def complete(command, **kwargs):
        calls.append(command)
        output = tmp_path / "runs/candidate"
        (output / "weights.pt").write_bytes(b"candidate checkpoint")
        candidate = {**metadata, "task": "detection", "architecture": "yolo_rai",
                     "config": {"epochs": 3, "image_size": 384, "batch_size": 8},
                     "initialization": job["parent_sha256"],
                     "weights_sha256": runner.digest(output / "weights.pt")}
        candidate["metrics"] = {"map50": 0.3, "per_class": []}
        (output / "manifest.json").write_text(json.dumps(candidate))
        return SimpleNamespace(pid=123, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", complete)
    result = runner.run(plan)
    assert result["state"] == "completed"
    assert calls[0][-2:] == ["--resume-checkpoint", str(checkpoint)]
    assert result["jobs"]["candidate"]["resumed"] is True


def test_campaign_rejects_plan_mutation_without_rewriting_journal(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    journal = plan.parent / "status.json"
    original = json.dumps({"plan_sha256": "other", "state": "completed"})
    journal.write_text(original)
    with pytest.raises(ValueError, match="plan changed"):
        runner.run(plan)
    assert journal.read_text() == original


def test_completed_candidate_must_match_the_planned_recipe():
    job = {"pipeline": "scratch", "options": {"epochs": 10, "learning-rate": 0.0005}}
    candidate = {"task": "detection", "architecture": "sva-detector", "config": {"epochs": 3, "learning_rate": 0.0005}}
    with pytest.raises(ValueError, match="recipe differs"):
        runner.verify_recipe(job, candidate)


def test_expansion_preserves_parent_roles_including_resized_source_hashes():
    previous = {"samples": [{"group": "camera-a", "split": "train", "sha256": "original"}]}
    expanded = {"samples": [{"group": "camera-b", "split": "test", "sha256": "resized", "source_sha256": "original"}]}
    with pytest.raises(ValueError, match="source content"):
        runner.verify_expansion(previous, expanded)
    expanded["samples"][0] = {"group": "camera-a", "split": "validation", "sha256": "different"}
    with pytest.raises(ValueError, match="source group"):
        runner.verify_expansion(previous, expanded)
    expanded["samples"][0]["split"] = "train"
    runner.verify_expansion(previous, expanded)


def test_yolo_resolution_and_batch_aliases_match_recorded_recipe():
    job = {"pipeline": "yolo", "options": {"imgsz": 512, "batch": 4}}
    candidate = {"task": "detection", "architecture": "yolo_rai", "config": {"image_size": 512, "batch_size": 4}}
    runner.verify_recipe(job, candidate)


def test_campaign_disk_reserve_prevents_launch(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    data = json.loads(plan.read_text())
    data["minimum_free_disk_bytes"] = 100
    plan.write_text(json.dumps(data))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=99))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    result = runner.run(plan)
    assert result["state"] == "resource_limited"
    assert result["jobs"]["candidate"]["state"] == "not_started_resource_limit"
    assert "disk reserve" in result["jobs"]["candidate"]["error"]


def test_training_cuda_requirement_rejects_cpu_fallback(monkeypatch):
    import torch
    from surveilx.accelerators import torch_device
    monkeypatch.setenv("SURVEILX_REQUIRE_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        torch_device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_device() == "cuda"


def test_campaign_checks_commit_before_starting_child(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    data = json.loads(plan.read_text())
    data["minimum_available_commit_bytes"] = 100
    plan.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "available_commit_bytes", lambda: 99)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    result = runner.run(plan)
    assert result["state"] == "resource_limited"
    assert "commit" in result["error"]


def test_child_resource_failure_halts_queue_even_after_memory_recovers(tmp_path, monkeypatch):
    plan, job, _ = fixture(tmp_path, monkeypatch)
    plan.write_text(json.dumps({'name': 'campaign', 'jobs': [job, {**job, 'version': 'second'}]}))
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        kwargs['stdout'].write('RuntimeError: Training resource guard: Windows commit reserve reached\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(pid=123, wait=lambda: 1)
    monkeypatch.setattr(runner.subprocess, 'Popen', fail)
    result = runner.run(plan)
    assert len(calls) == 1
    assert result['state'] == 'resource_limited'
    assert result['jobs']['candidate']['state'] == 'failed_resource_limit'
    assert 'second' not in result['jobs']


def test_old_resource_error_becomes_history_on_clean_continuation(tmp_path, monkeypatch):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    original = {"plan_sha256": runner.digest(plan), "jobs": {}, "state": "resource_limited", "error": "old disk pressure"}
    (plan.parent / "status.json").write_text(json.dumps(original))
    # Preserve a synthetic partial so this unit test cannot launch any process.
    out = tmp_path / "runs/candidate"
    out.mkdir(parents=True)
    (out / "weights.pt").write_bytes(b"partial")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No training permitted"))
    result = runner.run(plan)
    assert "error" not in result
    assert result["historical_errors"][0]["error"] == "old disk pressure"
