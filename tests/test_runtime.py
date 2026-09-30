import json
import time
from types import SimpleNamespace

from surveilx.controller import Candidate, Scheduler


def test_slow_camera_reserves_bounded_credit_and_impossible_cost_is_reported():
    scheduler = Scheduler()
    candidates = [Candidate("slow", 1, 8, 250), Candidate("fast", 5, 1, 80)]
    assert scheduler.allocate(candidates, 100, 5)["reserved_for"] == "slow"
    assert scheduler.allocate(candidates, 100, 5)["selected"] == []
    assert scheduler.allocate(candidates, 100, 5)["selected"] == ["slow"]
    result = Scheduler().allocate([Candidate("impossible", 1, 8, 600)], 100, 5)
    assert result["deferred"][0]["infeasible"] is True
    assert result["selected"] == []


def make_capture(client):
    from surveilx import api
    from surveilx.database import Camera, transaction
    from surveilx.security import cipher
    from surveilx.vision import generated_frame

    with transaction() as session:
        camera = Camera(
            name="Runtime fixture", source_cipher=cipher().encrypt(b"demo://0").decode(), zones=[[0, 0, 1, 1]]
        )
        session.add(camera)
        session.flush()
        camera_id = camera.id
    api.runtime.sync_cameras()
    capture = api.runtime.cameras[camera_id]
    capture.frame = generated_frame(0, 1)
    capture.sequence = 1
    capture.timestamp = time.time()
    return api.runtime, capture


def test_no_worker_leak_duplicate_frame_or_stale_secondary_result(client):
    runtime, capture = make_capture(client)
    assert capture.thread.ident is None
    runtime.infer(capture)
    assert capture.outputs and capture.outputs[0]["label"] == "synthetic_entity"
    count = len(runtime.latencies)
    runtime.infer(capture)
    assert len(runtime.latencies) == count
    capture.secondary_result = {"decision": "old_result"}
    capture.sequence += 1
    runtime.infer(capture)
    assert capture.secondary_result is None


def test_secondary_failure_keeps_primary_detections(client):
    runtime, capture = make_capture(client)

    def fail(*args):
        raise RuntimeError("Expert failure")

    runtime.sva = SimpleNamespace(infer=fail)
    capture.candidate_since = time.monotonic()
    runtime.infer(capture)
    assert capture.outputs
    assert capture.secondary_result["decision"] == "abstain"
    assert "Secondary expert failed" in runtime.error


def test_real_detector_is_restricted_to_its_evaluated_domain(client):
    runtime, capture = make_capture(client)
    capture.synthetic = False
    calls = []
    runtime.learned_detector = SimpleNamespace(
        manifest={"synthetic": False, "domain": "pedestrian"},
        infer=lambda *args: calls.append("learned") or [],
        name="learned",
        version="fixture",
        calibrated=False,
    )
    runtime.baseline = SimpleNamespace(
        infer=lambda *args: [], name="baseline", version="fixture", calibrated=False
    )
    runtime.infer(capture)
    assert not calls
    capture.environment = "pedestrian"
    capture.sequence += 1
    runtime.infer(capture)
    assert calls == ["learned"]


def test_retention_deletes_original_and_grouped_bundles(client):
    runtime, capture = make_capture(client)
    from surveilx.config import settings
    from surveilx.database import Incident, transaction
    from surveilx.evidence import save_bundle
    from surveilx.incidents import create_incident

    keys = [save_bundle([], {"synthetic": True}) for _ in range(2)]
    with transaction() as session:
        incident = create_incident(session, capture.id, keys[0], {"synthetic": True}, True)
        incident_id = incident.id
    with transaction() as session:
        incident = create_incident(session, capture.id, keys[1], {"synthetic": True}, True)
        incident.created = time.time() - (settings.retention_days + 1) * 86400
    runtime.retention()
    assert all(not (settings.data_dir / "evidence" / key).exists() for key in keys)
    with transaction() as session:
        incident = session.get(Incident, incident_id)
        assert incident.evidence_key is None
        assert incident.details["additional_evidence"] == []


def test_bad_spool_file_does_not_block_later_events(client):
    runtime, capture = make_capture(client)
    from surveilx.config import settings
    from surveilx.database import Incident, transaction
    from surveilx.security import cipher
    from sqlalchemy import select

    spool = settings.data_dir / "spool"
    spool.mkdir()
    (spool / "bad.event").write_bytes(b"damaged")
    payload = {"camera_id": capture.id, "evidence_key": "stable.bundle", "details": {}, "synthetic": True}
    (spool / "valid.event").write_bytes(cipher().encrypt(json.dumps(payload).encode()))
    runtime.flush_spool()
    assert (spool / "bad.event").exists()
    assert not (spool / "valid.event").exists()
    with transaction() as session:
        assert session.scalar(select(Incident).where(Incident.camera_id == capture.id))


def test_job_failure_preserves_metadata_releases_lock_and_restart_is_explicit(client, monkeypatch):
    from surveilx import jobs
    from surveilx.config import settings
    from surveilx.database import Record, transaction

    directory = settings.data_dir / "datasets" / "fixture"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"task": "detection"}))
    with transaction() as session:
        session.add(
            Record(
                id="failure",
                kind="experiment",
                payload={"state": "queued", "dataset": "fixture", "epochs": 2},
            )
        )
        session.add(
            Record(id="interrupted", kind="experiment", payload={"state": "running", "dataset": "fixture"})
        )
    called = []

    def subprocess_run(command, **kwargs):
        called.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(jobs.subprocess, "run", subprocess_run)
    manager = jobs.TrainingJobs()
    manager.running = True
    manager.run("failure", "fixture", 2, "test")
    assert manager.running is False
    assert "training.detection_pipeline" in called[0]
    manager.recover_interrupted()
    with transaction() as session:
        failed = session.get(Record, "failure").payload
        assert failed["state"] == "failed" and failed["dataset"] == "fixture" and failed["epochs"] == 2
        assert "finished" in failed
        assert session.get(Record, "interrupted").payload["state"] == "interrupted"
