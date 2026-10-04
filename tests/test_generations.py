import hashlib
import json
from types import SimpleNamespace

from surveilx.generations import queue_matches, read_generation, read_queue, runner_matches


def test_runner_identity_matches_module_and_generation_not_pid_alone(tmp_path):
    directory = tmp_path / "g2"
    event = SimpleNamespace(cmdline=lambda: ["python", "-m", "scripts.train_event_generation", "dataset", "g2"])
    assert runner_matches(event, directory)
    wrong = SimpleNamespace(cmdline=lambda: ["python", "-m", "scripts.train_event_generation", "dataset", "other"])
    assert not runner_matches(wrong, directory)
    unrelated = SimpleNamespace(cmdline=lambda: ["python", "-c", "scripts.train_event_generation g2"])
    assert not runner_matches(unrelated, directory)
    campaign = SimpleNamespace(cmdline=lambda: ["python", "-m", "scripts.train_accuracy_campaign", "g2/plan.json"],
                               cwd=lambda: str(tmp_path))
    assert runner_matches(campaign, directory)


def test_campaign_progress_exposes_pending_jobs_comparison_and_yolo_epochs(tmp_path, monkeypatch):
    import psutil

    directory = tmp_path / "generations/g2"
    directory.mkdir(parents=True)
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps({"jobs": [
        {"version": "child", "pipeline": "yolo", "options": {"epochs": 8}},
        {"version": "later", "pipeline": "scratch", "options": {"epochs": 8}},
    ]}))
    results = tmp_path / "runs/child/fit/results.csv"
    results.parent.mkdir(parents=True)
    results.write_text("epoch,loss\n1,0.4\n2,0.3\n")
    path = directory / "status.json"
    original = json.dumps({"state": "running", "pid": 100, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                           "jobs": {"child": {"state": "running", "comparison": {"delta": -0.1}}}})
    path.write_text(original)
    process = SimpleNamespace(cmdline=lambda: ["python", "-m", "scripts.train_accuracy_campaign", str(plan_path)])
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    result = read_generation(path, tmp_path)
    assert result["state"] == "running"
    assert result["jobs"][0]["completed_epochs"] == 2
    assert result["jobs"][0]["comparison"]["delta"] == -0.1
    assert result["jobs"][1]["state"] == "pending"
    assert path.read_text() == original


def test_event_journal_has_visible_job_history(tmp_path):
    directory = tmp_path / "generations/event"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"state": "completed", "version": "fall-scene"}))
    run = tmp_path / "runs/fall-scene"
    run.mkdir(parents=True)
    (run / "history.json").write_text('[{"epoch": 1}]')
    result = read_generation(directory / "status.json", tmp_path)
    assert result["jobs"][0]["version"] == "fall-scene"
    assert result["jobs"][0]["completed_epochs"] == 1


def test_optimized_profiles_are_visible(tmp_path):
    directory = tmp_path / "generations/profiles"
    directory.mkdir(parents=True)
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps({"jobs": [{"version": "optimized"}]}))
    profiles = [{"tier": "economy", "image_size": 224, "validation_map50": 0.52,
                 "latency_ms": 14.2, "test_metrics": {"map50": 0.55}}]
    journal = {"state": "completed", "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
               "jobs": {"optimized": {"state": "completed", "profiles": profiles}}}
    path = directory / "status.json"
    path.write_text(json.dumps(journal))
    result = read_generation(path, tmp_path)
    assert result["jobs"][0]["profiles"] == profiles


def test_waiting_queue_is_visible_and_process_verified(tmp_path, monkeypatch):
    directory = tmp_path / "generations/profiles"
    directory.mkdir(parents=True)
    path = directory / "queue.json"
    path.write_text(json.dumps({"state": "waiting", "pid": 123, "ppe_state": "waiting", "ppe_retry_state": "planned"}))
    process = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "scripts.queue_model_optimization", str(directory)],
        cwd=lambda: str(tmp_path),
    )
    monkeypatch.setattr("surveilx.generations.psutil.Process", lambda pid: process)
    assert queue_matches(process, directory)
    result = read_queue(path)
    assert result["state"] == "waiting"
    assert result["config"]["ppe_state"] == "waiting"
    assert result["config"]["ppe_retry_state"] == "planned"


def test_accuracy_retry_queue_identity_is_visible(tmp_path, monkeypatch):
    directory = tmp_path / "generations/retry"
    directory.mkdir(parents=True)
    path = directory / "queue.json"
    path.write_text(json.dumps({"state": "waiting", "pid": 321, "predecessor_state": "running"}))
    process = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "scripts.queue_accuracy_retry", str(directory), "original/status.json"],
        cwd=lambda: str(tmp_path),
    )
    monkeypatch.setattr("surveilx.generations.psutil.Process", lambda pid: process)
    assert queue_matches(process, directory)
    result = read_queue(path)
    assert result["state"] == "waiting"
    assert result["config"]["predecessor_state"] == "running"
