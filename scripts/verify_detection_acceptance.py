"""Smoke-test detector acceptance inference; the reused benchmark is NOT independent acceptance."""

import json
from pathlib import Path

from training.acceptance import predict_artifact, verify_independence
from training.datasets import validate_manifest


def main():
    dataset_path = Path("data/datasets/pennfudan/manifest.json")
    dataset, _ = validate_manifest(dataset_path)
    try:
        verify_independence(dataset_path, [dataset_path])
    except ValueError as exc:
        leakage_error = str(exc)
    else:
        raise AssertionError("Reused training/evaluation dataset must fail independence")
    results = []
    for version in ["scratch-pennfudan-v1", "yolo-rai-pennfudan-v1"]:
        manifest, metrics = predict_artifact(Path("data/runs") / version, dataset_path, dataset)
        results.append({"version": version, "weights_sha256": manifest["weights_sha256"], "metrics": metrics})
        print(version, metrics["map50"], flush=True)
    report = {"scope": "Inference integration only; reused Penn-Fudan test data cannot authorize production",
              "independence_rejection": leakage_error, "deployment_eligible": False, "results": results}
    Path("reports/acceptance-detector-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
