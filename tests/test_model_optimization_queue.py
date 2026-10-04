import hashlib
import json

from scripts.queue_model_optimization import choose_source, resolution_candidates, retry_state


def test_resolution_candidates_are_bounded_unique_multiples_of_32():
    assert resolution_candidates(256) == [160, 224, 256]
    assert resolution_candidates(384) == [256, 320, 384]
    assert resolution_candidates(512) == [320, 416, 512]
    for size in (128, 192, 256, 384, 512):
        values = resolution_candidates(size)
        assert values == sorted(set(values))
        assert all(value >= 128 and value % 32 == 0 for value in values)


def write_artifact(root, version, *, calibrated=True, correct_hash=True):
    run = root / "runs" / version
    run.mkdir(parents=True)
    weights = b"complete checkpoint"
    (run / "weights.pt").write_bytes(weights)
    checkpoint_hash = hashlib.sha256(weights).hexdigest()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "task": "detection",
                "calibrated": calibrated,
                "calibration": {"status": "fitted" if calibrated else "not_fitted"},
                "weights_sha256": checkpoint_hash if correct_hash else "0" * 64,
            }
        )
    )


def test_source_selection_rejects_partial_or_corrupt_preferred(monkeypatch, tmp_path):
    from scripts import queue_model_optimization as queue

    monkeypatch.setattr(queue.settings, "data_dir", tmp_path)
    write_artifact(tmp_path, "preferred", correct_hash=False)
    write_artifact(tmp_path, "fallback")
    assert choose_source("preferred", "fallback") == "fallback"


def test_retry_state_distinguishes_planned_from_absent(tmp_path):
    directory = tmp_path / "generations/retry"
    directory.mkdir(parents=True)
    assert retry_state(directory) == "absent"
    (directory / "plan.json").write_text("{}")
    assert retry_state(directory) == "planned"
    (directory / "queue.json").write_text(json.dumps({"state": "completed"}))
    assert retry_state(directory) == "completed"
