import pytest

from surveilx.policies import PolicyEngine, SitePolicy
from test_api import login


def detection(label="person", box=None, track_id=1):
    return {"label": label, "box": box or [0.1, 0.1, 0.5, 0.9], "score": 0.9, "track_id": track_id}


def engine(kind="object_presence", **extra):
    return PolicyEngine({"rules": [{"id": "test", "kind": kind, "labels": ["person"], **extra}]})


def test_persistence_gap_cooldown_and_model_isolation():
    rule = engine()
    assert not rule.evaluate([detection()], 0, "a")
    assert not rule.evaluate([detection()], 1, "a")
    assert not rule.evaluate([detection()], 2, "b")
    assert len(rule.evaluate([detection()], 2, "a")) == 1
    assert not rule.evaluate([detection()], 3, "a")
    assert not rule.evaluate([detection()], 70, "a")  # stale continuity resets
    rule.evaluate([], 71, "a")
    assert not rule.evaluate([detection()], 72, "a")


def test_wrong_way_uses_configured_direction_and_temporal_displacement():
    rule = engine("wrong_way", direction=[1, 0], min_displacement=0.1)
    for index in range(2):
        assert not rule.evaluate(
            [detection(box=[0.6 - index * 0.1, 0.1, 0.8 - index * 0.1, 0.8])], index, "a"
        )
    assert rule.evaluate([detection(box=[0.4, 0.1, 0.6, 0.8])], 2, "a")[0]["kind"] == "wrong_way"


def test_helmet_rule_abstains_without_visible_head_or_with_helmet():
    rule = engine("possible_missing_helmet")
    person, head = detection(), detection("head", [0.2, 0.15, 0.3, 0.25], 2)
    for now in range(4):
        assert not rule.evaluate([person], now, "a")
    for now in (5, 6):
        assert not rule.evaluate([person, head], now, "a")
    assert rule.evaluate([person, head], 7, "a")
    rule = engine("possible_missing_helmet")
    for now in range(4):
        assert not rule.evaluate([person, head, detection("helmet", [0.2, 0.12, 0.3, 0.23], 3)], now, "a")


def test_policy_validation_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        SitePolicy.model_validate(
            {"rules": [{"id": "x", "kind": "wrong_way", "labels": ["car"], "direction": [0, 0]}]}
        )


def test_policy_api_persists_and_camera_loads_rules(client):
    login(client)
    camera = client.post("/api/cameras", json={"name": "Rules", "source": "demo://0"}).json()
    path = f"/api/cameras/{camera['id']}/policies"
    body = {"rules": [{"id": "fire", "kind": "object_presence", "labels": ["fire", "smoke"]}]}
    assert client.put(path, json=body).status_code == 200
    assert client.get(path).json()["rules"][0]["id"] == "fire"
    from surveilx import api

    assert api.runtime.cameras[camera["id"]].policy_engine.policy.rules[0].id == "fire"


def test_scene_event_policy_requires_calibration_persistence_and_model_isolation():
    engine = PolicyEngine(
        {
            "rules": [
                {
                    "id": "fall",
                    "kind": "scene_event",
                    "labels": ["fall"],
                    "confidence": 0.7,
                    "duration": 1,
                    "min_observations": 2,
                }
            ]
        }
    )
    evidence = {
        "decision": "experimental_evidence",
        "calibrated": True,
        "classes": ["normal", "fall"],
        "probabilities": [0.1, 0.9],
    }
    assert engine.evaluate_event(evidence, 0, ("event:fall", "v1")) == []
    result = engine.evaluate_event(evidence, 1, ("event:fall", "v1"))
    assert result[0]["label"] == "fall" and result[0]["score"] == 0.9
    assert engine.evaluate_event(evidence, 2, ("event:fall", "v2")) == []
    assert engine.evaluate_event({**evidence, "calibrated": False}, 3, ("event:fall", "v2")) == []
