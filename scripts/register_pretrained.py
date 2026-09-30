"""Register locally installed broad YOLO weights as uncalibrated domain-scoped candidates."""

import argparse
import json
from pathlib import Path
import re
import shutil

from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import ModelVersion, audit, transaction
from training.datasets import digest


def register(source, domain):
    from ultralytics import YOLO

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", domain):
        raise ValueError("Use a simple domain identifier")
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("An existing local checkpoint is required")
    checksum = digest(source)
    version = f"yolo11n-general-{domain}-v1"
    output = settings.data_dir / "runs" / version
    with transaction() as session:
        existing = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
        if existing:
            if (
                existing.manifest.get("weights_sha256") != checksum
                or digest(output / "weights.pt") != checksum
            ):
                raise ValueError("Existing pretrained registration differs; preserve it and inspect")
            return existing.id
    model = YOLO(str(source))
    classes = [model.names[i] for i in range(len(model.names))]
    if len(classes) != 80 or not {"person", "car", "chair"} <= set(classes):
        raise ValueError("Expected the broad COCO checkpoint, not a person-only fine-tuned model")
    metadata = {
        "schema_version": 1,
        "name": f"Pretrained general objects ({domain})",
        "task": "detection",
        "architecture": "yolo_pretrained",
        "training_status": "external_pretrained",
        "domain": domain,
        "classes": classes,
        "capability_ids": ["general_objects", "person_detection"],
        "synthetic": False,
        "calibrated": False,
        "calibration": {"status": "unfitted", "threshold": 0.25},
        "config": {"image_size": 320},
        "weights_sha256": checksum,
        "deployment_eligible": False,
        "license": "Ultralytics AGPL-3.0 / Enterprise terms; inherited model, not locally trained",
        "gate_reason": "No local domain calibration or independent acceptance; canary only",
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, output / "weights.pt")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with transaction() as session:
        row = ModelVersion(name=metadata["name"], version=version, manifest=metadata)
        session.add(row)
        session.flush()
        audit(session, "local-cli", "pretrained_candidate_registered", row.id, domain=domain)
        return row.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()
    print(register(args.source, args.domain))
