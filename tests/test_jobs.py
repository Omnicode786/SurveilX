import json

import pytest

from surveilx import jobs


def test_retention_report_requires_byte_identical_heldout_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.settings, "data_dir", tmp_path)
    samples = [
        {"file": "train.npz", "split": "train", "group": "a", "label": 0, "sha256": "1"},
        {"file": "test.npz", "split": "test", "group": "b", "label": 1, "sha256": "2"},
    ]
    base_path = tmp_path / "datasets/base/manifest.json"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(json.dumps({"samples": samples}))
    parent = tmp_path / "runs/parent"
    parent.mkdir(parents=True)
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": jobs.digest(base_path),
                "weights_sha256": "parent-weights",
                "metrics": {"accuracy": 0.8},
            }
        )
    )
    current = {
        "samples": [*samples, {"file": "new.npz", "split": "train", "group": "c", "label": 1}],
        "adaptation": {"base_dataset": "base"},
    }
    report = jobs.retention_report({"task": "event", "metrics": {"accuracy": 0.75}}, current, parent)
    assert report["delta"] == pytest.approx(-0.05)
    current["samples"][1]["sha256"] = "changed"
    with pytest.raises(ValueError, match="samples changed"):
        jobs.retention_report({"task": "event", "metrics": {"accuracy": 0.75}}, current, parent)
