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


def test_vest_rule_requires_visible_unambiguous_torso_and_abstains_with_vest():
    rule = engine("possible_missing_vest")
    person = detection()
    torso = detection("torso", [0.18, 0.3, 0.42, 0.65], 2)
    for now in range(4):
        assert not rule.evaluate([person], now, "a")
    for now in (5, 6):
        assert not rule.evaluate([person, torso], now, "a")
    result = rule.evaluate([person, torso], 7, "a")
    assert result[0]["kind"] == "possible_missing_vest"

    rule = engine("possible_missing_vest")
    vest = detection("safety_vest", [0.2, 0.35, 0.4, 0.6], 3)
    for now in range(4):
        assert not rule.evaluate([person, torso, vest], now, "a")


def test_configured_proximity_persists_a_specific_tracked_pair():
    rule = engine(
        "configured_proximity",
        labels=["person"],
        secondary_labels=["forklift"],
        max_distance=0.2,
    )
    person = detection(box=[0.1, 0.1, 0.3, 0.8], track_id=1)
    far = detection("forklift", [0.7, 0.2, 0.9, 0.8], 2)
    near = detection("forklift", [0.25, 0.2, 0.45, 0.8], 2)
    assert not rule.evaluate([person, far], 0, "a")
    assert not rule.evaluate([person, near], 1, "a")
    assert not rule.evaluate([person, near], 2, "a")
    result = rule.evaluate([person, near], 3, "a")
    assert result[0]["related_label"] == "forklift"
    assert result[0]["related_track_id"] == 2
    assert result[0]["normalized_center_distance"] < 0.2


def test_calibrated_speed_uses_ground_plane_transform_and_persistence():
    rule = engine(
        "calibrated_speed",
        labels=["car"],
        calibration_id="south_gate_homography_v1",
        ground_plane_transform=[10, 0, 0, 0, 10, 0, 0, 0, 1],
        speed_mps=1.0,
        duration=1,
        min_observations=2,
    )
    assert not rule.evaluate([detection("car", [0.1, 0.1, 0.2, 0.2])], 0, "traffic")
    assert not rule.evaluate([detection("car", [0.3, 0.1, 0.4, 0.2])], 1, "traffic")
    result = rule.evaluate([detection("car", [0.5, 0.1, 0.6, 0.2])], 2, "traffic")
    assert result[0]["kind"] == "calibrated_speed"
    assert result[0]["metric_speed_mps"] == pytest.approx(2.0)
    assert result[0]["calibration_id"] == "south_gate_homography_v1"


def test_signal_stop_line_requires_configured_signal_and_crossing():
    rule = engine(
        "signal_stop_line",
        labels=["car"],
        secondary_labels=["red_signal"],
        zone=[0.5, 0, 1, 1],
        signal_zone=[0, 0, 0.3, 0.3],
        stop_line=[0.5, 0, 0.5, 1],
        duration=1,
        min_observations=2,
    )
    red = detection("red_signal", [0.05, 0.05, 0.15, 0.15], 20)
    assert not rule.evaluate([detection("car", [0.35, 0.2, 0.45, 0.6]), red], 0, "traffic")
    assert not rule.evaluate([detection("car", [0.5, 0.2, 0.6, 0.6]), red], 1, "traffic")
    result = rule.evaluate([detection("car", [0.6, 0.2, 0.7, 0.6]), red], 2, "traffic")
    assert result[0]["kind"] == "signal_stop_line"
    assert result[0]["signal_label"] == "red_signal"


def test_signal_stop_line_abstains_without_signal_state():
    rule = engine(
        "signal_stop_line",
        labels=["car"],
        secondary_labels=["red_signal"],
        zone=[0.5, 0, 1, 1],
        stop_line=[0.5, 0, 0.5, 1],
    )
    assert not rule.evaluate([detection("car", [0.35, 0.2, 0.45, 0.6])], 0, "traffic")
    assert not rule.evaluate([detection("car", [0.6, 0.2, 0.7, 0.6])], 1, "traffic")


def test_machine_state_requires_nearby_configured_state_label():
    rule = engine(
        "machine_state",
        labels=["press"],
        secondary_labels=["unguarded_running"],
        max_distance=0.3,
    )
    press = detection("press", [0.2, 0.2, 0.5, 0.7], 7)
    far_state = detection("unguarded_running", [0.8, 0.2, 0.9, 0.4], 8)
    near_state = detection("unguarded_running", [0.45, 0.2, 0.55, 0.4], 8)
    assert not rule.evaluate([press, far_state], 0, "industrial")
    assert not rule.evaluate([press, near_state], 1, "industrial")
    assert not rule.evaluate([press, near_state], 2, "industrial")
    result = rule.evaluate([press, near_state], 3, "industrial")
    assert result[0]["kind"] == "machine_state"
    assert result[0]["related_label"] == "unguarded_running"


def test_blocked_exit_is_explicit_persistent_zone_review():
    rule = engine("blocked_exit", labels=["pallet"], zone=[0.1, 0.1, 0.5, 0.8])
    pallet = detection("pallet", [0.2, 0.2, 0.4, 0.6], 11)
    assert not rule.evaluate([pallet], 0, "warehouse")
    assert not rule.evaluate([pallet], 1, "warehouse")
    result = rule.evaluate([pallet], 2, "warehouse")
    assert result[0]["kind"] == "blocked_exit"


def test_policy_validation_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        SitePolicy.model_validate(
            {"rules": [{"id": "x", "kind": "wrong_way", "labels": ["car"], "direction": [0, 0]}]}
        )
    with pytest.raises(ValueError, match="calibration"):
        SitePolicy.model_validate({"rules": [{"id": "speed", "kind": "calibrated_speed", "labels": ["car"]}]})
    with pytest.raises(ValueError, match="secondary"):
        SitePolicy.model_validate({"rules": [{"id": "signal", "kind": "signal_stop_line", "labels": ["car"]}]})
    with pytest.raises(ValueError, match="Stop line"):
        SitePolicy.model_validate(
            {"rules": [{"id": "line", "kind": "signal_stop_line", "labels": ["car"], "secondary_labels": ["red"], "stop_line": [0.5, 0.5, 0.5, 0.5]}]}
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


def test_scene_event_resets_on_abstention_label_change_and_invalid_probabilities():
    policy = engine("scene_event", labels=["fall", "fighting"], duration=1, min_observations=2)
    def evidence(scores):
        return {"decision": "experimental_evidence", "calibrated": True,
                "classes": ["normal", "fall", "fighting"], "probabilities": scores}
    fall = evidence([0.1, 0.8, 0.1])
    fight = evidence([0.1, 0.1, 0.8])
    assert not policy.evaluate_event(fall, 0, "a")
    assert not policy.evaluate_event(None, 0.5, "a")
    assert not policy.evaluate_event(fall, 1, "a")
    assert not policy.evaluate_event(fight, 2, "a")
    assert not policy.evaluate_event(fight, 2, "a")
    assert not policy.evaluate_event(evidence([0, 0, float("inf")]), 2.5, "a")
    assert not policy.evaluate_event(fight, 3, "a")
    result = policy.evaluate_event(fight, 4, "a")
    assert result[0]["label"] == "fighting" and result[0]["observations"] == 2


def test_scene_rule_cannot_be_triggered_by_same_named_detection_box():
    policy = engine("scene_event", labels=["fall"])
    for now in range(5):
        assert not policy.evaluate([detection("fall", [0.1, 0.1, 0.8, 0.8])], now, "a")


def test_stop_line_does_not_extend_outside_configured_segment():
    from surveilx.policies import crosses_segment
    assert crosses_segment((0.4, 0.5), (0.6, 0.5), [0.5, 0.3, 0.5, 0.7])
    assert not crosses_segment((0.4, 0.9), (0.6, 0.9), [0.5, 0.3, 0.5, 0.7])
    assert not crosses_segment((0.5, 0.5), (0.6, 0.5), [0.5, 0.3, 0.5, 0.7])


def test_duplicate_observations_and_changed_machine_state_do_not_accumulate():
    policy = engine("object_presence", labels=["fire"], duration=1, min_observations=3)
    item = detection("fire", [0.1, 0.1, 0.6, 0.6])
    assert not policy.evaluate([item], 0, "a")
    assert not policy.evaluate([item], 1, "a")
    assert not policy.evaluate([item], 1, "a")
    assert policy.evaluate([item], 2, "a")[0]["observations"] == 3
    policy = engine("machine_state", labels=["press"], secondary_labels=["open", "unguarded"],
                    max_distance=0.5, duration=1, min_observations=2)
    press = detection("press", [0.2, 0.2, 0.5, 0.7], 7)
    state = detection("open", [0.3, 0.3, 0.4, 0.4], 8)
    assert not policy.evaluate([press, state], 0, "a")
    assert not policy.evaluate([press, {**state, "label": "unguarded"}], 1, "a")
    assert policy.evaluate([press, {**state, "label": "unguarded"}], 2, "a")
