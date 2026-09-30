import json
from types import SimpleNamespace

from scripts import train_event_generation as runner


def test_event_generation_registers_and_reuses_complete_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path)
    dataset = tmp_path / "datasets/fixture"
    dataset.mkdir(parents=True)
    manifest = dataset / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(runner, "validate_manifest", lambda path: ({"task": "event"}, {"train": 10}))
    monkeypatch.setattr(runner, "register", lambda version: "id-" + version)
    commands = []

    def complete(command, **kwargs):
        commands.append(command)
        output = tmp_path / "runs/g1-scene"
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"dataset_sha256": runner.digest(manifest)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", complete)
    assert runner.run("fixture", "g1", epochs=1)["state"] == "completed"
    assert runner.run("fixture", "g1", epochs=1)["model_id"] == "id-g1-scene"
    assert len(commands) == 1
