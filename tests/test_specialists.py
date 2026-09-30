from types import SimpleNamespace

from test_api import login
from test_runtime import make_capture


def test_named_slots_restore_and_rollback_without_replacing_other_specialists(client, monkeypatch):
    from surveilx import api
    from surveilx.database import ModelVersion, transaction
    from surveilx.runtime import Runtime

    login(client)
    monkeypatch.setattr(
        Runtime,
        "prepare_model",
        lambda self, model: SimpleNamespace(version=model.version, manifest=model.manifest),
    )
    with transaction() as session:
        models = [
            ModelVersion(name=name, version=name, manifest={"task": "detection", "domain": name})
            for name in ("fire", "ppe", "fire2")
        ]
        session.add_all(models)
        session.flush()
        first, second, third = [model.id for model in models]
    for model_id, slot in ((first, "fire"), (second, "ppe"), (third, "fire")):
        assert client.post(f"/api/models/{model_id}/deploy", json={"slot": slot}).status_code == 200
    assert client.get(f"/api/models/{second}").json()["stage"] == "canary"
    assert client.post(f"/api/models/{second}/deploy", json={"slot": "duplicate"}).status_code == 409
    restored = Runtime()
    restored.restore_models()
    assert restored.active_models["detection:fire"] == third
    assert restored.active_models["detection:ppe"] == second
    assert client.post(f"/api/models/{third}/rollback").status_code == 200
    assert api.runtime.active_models["detection:fire"] == first
    assert api.runtime.active_models["detection:ppe"] == second


def test_specialist_round_robin_is_domain_scoped_and_keeps_distinct_trackers(client):
    runtime, capture = make_capture(client)
    calls = []
    for slot, domain in (("a", capture.environment), ("b", capture.environment), ("other", "unrelated")):
        expert = SimpleNamespace(
            name=slot,
            version=slot,
            calibrated=False,
            manifest={"synthetic": True, "domain": domain},
            infer=lambda *args, name=slot: calls.append(name) or [],
        )
        runtime.install_model(f"detection:{slot}", expert, slot)
    runtime.infer(capture)
    first_tracker = capture.tracker
    capture.sequence += 1
    runtime.infer(capture)
    assert capture.tracker is not first_tracker
    capture.sequence += 1
    runtime.infer(capture)
    assert calls == ["a", "b", "a"]
    assert capture.tracker is first_tracker
