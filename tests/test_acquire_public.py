from pathlib import Path

import pytest

from scripts import acquire_public


def test_retry_resumes_only_when_partial_grows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acquire_public.time, "sleep", lambda _seconds: None)
    calls = []

    def truncated(_name):
        calls.append(True)
        partial = Path("data/downloads/dangerous-items/dataset.zip.part")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(partial.read_bytes() + b"chunk" if partial.exists() else b"chunk")
        if len(calls) < 3:
            raise ValueError("Incomplete archive; rerun to resume")

    monkeypatch.setattr(acquire_public, "acquire", truncated)
    acquire_public.acquire_with_retries("dangerous-items", attempts=3)
    assert len(calls) == 3


def test_retry_preserves_non_truncation_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def invalid(_name):
        raise ValueError("Remote archive changed during resumed acquisition")

    monkeypatch.setattr(acquire_public, "acquire", invalid)
    with pytest.raises(ValueError, match="Remote archive changed"):
        acquire_public.acquire_with_retries("dangerous-items")
