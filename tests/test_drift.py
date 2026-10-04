import time

from surveilx.drift import analyze_decisions, jensen_shannon


def decision(index, *, camera="camera-1", motion=0.01, brightness=0.5, latency=40, confidence=0.8):
    return {
        "timestamp": time.time() + index,
        "camera_id": camera,
        "expert_slot": "detection:fixture",
        "model_version": "fixture-v1",
        "state": {"motion": motion, "brightness": brightness},
        "latency_ms": latency,
        "result": [{"confidence": confidence}],
    }


def test_jensen_shannon_is_bounded_and_symmetric():
    bins = [-1, 0.5, 1.5, 3]
    first, second = [0, 0, 0, 0], [1, 1, 1, 1]
    forward = jensen_shannon(first, second, bins)
    assert 0 <= forward <= 1
    assert forward == jensen_shannon(second, first, bins)


def test_monitor_collects_before_non_overlapping_windows_exist():
    result = analyze_decisions([decision(i) for i in range(20)])
    assert result["state"] == "collecting"
    assert result["streams"][0]["required"] == 90


def test_large_multi_feature_shift_requests_review():
    reference = [decision(i) for i in range(60)]
    recent = [
        decision(i + 60, motion=0.8, brightness=0.95, latency=600, confidence=0.1)
        for i in range(30)
    ]
    result = analyze_decisions(reference + recent)
    assert result["state"] == "review_required"
    shifted = result["streams"][0]["shifted_features"]
    assert {"motion", "brightness", "latency_ms", "mean_confidence"}.issubset(shifted)
    assert "predictions are never converted to labels" in result["interpretation"]


def test_stable_stream_does_not_trigger_review():
    rows = [decision(i, motion=(i % 5) / 100, latency=40 + i % 3) for i in range(90)]
    result = analyze_decisions(rows)
    assert result["state"] == "stable"
    assert result["streams"][0]["shifted_features"] == []


def test_missing_features_do_not_report_false_stability():
    rows = [{"timestamp": i, "camera_id": "camera", "action": "unknown"} for i in range(90)]
    result = analyze_decisions(rows)
    assert result["state"] == "collecting"
    assert {row["feature"] for row in result["streams"][0]["features"]} == {
        "detection_count",
        "mean_confidence",
    }


def test_drift_endpoint_requires_research_access(client):
    client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"})
    response = client.get("/api/adaptation/drift")
    assert response.status_code == 200
    assert response.json()["state"] in {"collecting", "stable", "review_required"}
