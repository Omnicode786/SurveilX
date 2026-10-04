import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import surveilx.config
    import surveilx.database

    monkeypatch.setattr(surveilx.config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(surveilx.config.settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(surveilx.config.settings, "admin_password", "test-password-long")
    monkeypatch.setattr(surveilx.config.settings, "start_workers", False)
    monkeypatch.setattr(surveilx.config.settings, "demo", False)
    monkeypatch.setattr(surveilx.config.settings, "detector_path", "")
    monkeypatch.setattr(surveilx.config.settings, "training_python", "")
    monkeypatch.setattr(surveilx.config.settings, "training_batch_size", None)
    monkeypatch.setattr(surveilx.config.settings, "training_yolo_batch_size", None)
    monkeypatch.setattr(surveilx.config.settings, "training_event_batch_size", None)
    importlib.reload(surveilx.database)
    for name in [
        "security",
        "incidents",
        "notifications",
        "jobs",
        "adaptation",
        "adaptation_worker",
        "acceptance",
        "runtime",
        "api",
    ]:
        module = importlib.import_module(f"surveilx.{name}")
        importlib.reload(module)
    import surveilx.api

    with TestClient(surveilx.api.app) as test_client:
        yield test_client
