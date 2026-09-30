import json
import shutil

import pytest

from test_adaptation import detection_fixture


def accepted_artifact(session, model, root, synthetic=False):
    from surveilx.database import Record
    from training.datasets import digest

    directory = root / "runs" / model.version
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"fixed-test-weights")
    metadata = {
        "domain": "fixture",
        "synthetic": synthetic,
        "weights_sha256": digest(directory / "weights.pt"),
    }
    (directory / "manifest.json").write_text(json.dumps(metadata))
    report = {
        "passed": True,
        "deployment_eligible": not synthetic,
        "synthetic": synthetic,
        "domain": "fixture",
        "dataset_sha256": "test-dataset",
        "weights_sha256": metadata["weights_sha256"],
        "artifact_manifest_sha256": digest(directory / "manifest.json"),
    }
    model.manifest = metadata
    report_id = f"accept-{model.id}"
    session.add(
        Record(
            id=report_id,
            kind="acceptance",
            payload={"state": "completed", "model_id": model.id, "report": report},
        )
    )
    return report_id


def test_independence_checks_groups_and_exact_content(tmp_path):
    from training.acceptance import verify_independence

    reference = detection_fixture(tmp_path)
    acceptance = detection_fixture(tmp_path, "independent", offset=3)
    assert verify_independence(acceptance, [reference])["domain"] == "fixture"
    with pytest.raises(ValueError, match="groups overlap"):
        verify_independence(reference, [reference])
    document = json.loads(acceptance.read_text())
    # Different group and filename still cannot hide an exact duplicate.
    shutil.copyfile(reference.parent / "0.png", acceptance.parent / "3.png")
    document["samples"][-1].pop("sha256")
    acceptance.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="content overlaps"):
        verify_independence(acceptance, [reference])


def test_mathematical_acceptance_bounds_and_baseline_regression():
    from surveilx.acceptance import AcceptancePolicy
    from training.acceptance import gate, wilson_lower

    assert wilson_lower(0, 0) == 0
    assert wilson_lower(0, 10) == pytest.approx(0)
    assert wilson_lower(10, 10) == pytest.approx(0.72246, abs=0.0001)
    assert wilson_lower(100, 100) > wilson_lower(10, 10)
    metrics = {
        "samples": 200,
        "independent_groups": 10,
        "per_class_support": [100, 100],
        "map50": 0.85,
        "recall_lower_95": 0.82,
        "precision_lower_95": 0.83,
        "ece": 0.02,
        "wall_ms_per_sample": 100,
    }
    policy = AcceptancePolicy().model_dump()
    assert gate(metrics, policy, "detection")["passed"]
    assert not gate(metrics, policy, "detection", {"map50": 0.95})["passed"]
    for field, value in [
        ("samples", 9),
        ("independent_groups", 1),
        ("per_class_support", [0, 100]),
        ("recall_lower_95", 0),
        ("ece", float("nan")),
        ("wall_ms_per_sample", float("inf")),
    ]:
        assert not gate({**metrics, field: value}, policy, "detection")["passed"]


def test_acceptance_approval_is_not_deployment_and_binds_calibration(client):
    from surveilx.acceptance import accept_report, verify_approved_artifact
    from surveilx.config import settings
    from surveilx.database import ModelVersion, transaction

    with transaction() as session:
        model = ModelVersion(name="fixture", version="fixture", manifest={})
        session.add(model)
        session.flush()
        model_id = model.id
        report_id = accepted_artifact(session, model, settings.data_dir)
    assert accept_report(report_id, "test-reviewer")["stage"] == "candidate"
    with transaction() as session:
        verify_approved_artifact(session.get(ModelVersion, model_id), session)
    path = settings.data_dir / "runs/fixture/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["calibration"] = {"threshold": 0.001}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="calibration changed"):
        accept_report(report_id, "test-reviewer")
    with transaction() as session, pytest.raises(ValueError, match="differs"):
        verify_approved_artifact(session.get(ModelVersion, model_id), session)


def test_synthetic_report_cannot_authorize_production(client):
    from surveilx.acceptance import accept_report
    from surveilx.config import settings
    from surveilx.database import ModelVersion, transaction

    with transaction() as session:
        model = ModelVersion(name="fixture", version="fixture", manifest={})
        session.add(model)
        session.flush()
        report_id = accepted_artifact(session, model, settings.data_dir, synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        accept_report(report_id, "test-reviewer")


def test_acceptance_follows_all_dataset_ancestors(client):
    from surveilx.acceptance import references_for
    from surveilx.config import settings
    from training.datasets import digest
    from types import SimpleNamespace

    original = detection_fixture(settings.data_dir)
    adapted = detection_fixture(settings.data_dir, "adapted", offset=3)
    metadata = json.loads(adapted.read_text())
    metadata["adaptation"] = {"base_sha256": digest(original)}
    adapted.write_text(json.dumps(metadata))
    model = SimpleNamespace(manifest={"dataset_sha256": digest(adapted)})
    assert len(references_for([model])) == 2
    original.write_text(original.read_text() + " ")
    with pytest.raises(ValueError, match="missing or modified"):
        references_for([model])
