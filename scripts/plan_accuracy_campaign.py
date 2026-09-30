"""Freeze the current trained model families and a CPU-sized continuation recipe."""

import argparse
import json
from pathlib import Path

from surveilx.config import settings
from training.datasets import digest


def create(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        raise ValueError("Plan already exists; resume it without regenerating")
    families = [
        ("urfall-development-v1", "urfall-g1-scene", "event", "fall"),
        ("generated-v1", "trajectory-v1", "event", "entity-v1"),
        ("generated-v2", "synthetic-candidate-v1", "event", "entity-v2"),
        ("sh17-development-v1", "sh17-g2-scratch", "scratch", "ppe-scratch"),
        ("sh17-development-v1", "sh17-g2-yolo-rai", "yolo", "ppe-yolo"),
        ("dangerous-items-development-v1", "dangerous-items-g1-scratch", "scratch", "weapons-scratch"),
        ("dangerous-items-development-v1", "dangerous-items-g1-yolo-rai", "yolo", "weapons-yolo"),
        ("dfire-development-v1", "dfire-g1-scratch", "scratch", "fire-scratch"),
        ("dfire-development-v1", "dfire-g1-yolo-rai", "yolo", "fire-yolo"),
        ("pennfudan", "scratch-pennfudan-v1", "scratch", "person-scratch"),
        ("pennfudan", "yolo-rai-pennfudan-v1", "yolo", "person-yolo"),
        ("pennfudan", "yolo-baseline-pennfudan-v1", "yolo", "person-baseline"),
    ]
    jobs = []
    for dataset, parent, pipeline, suffix in families:
        options = {"epochs": 60 if pipeline == "event" else 8, "threads": 3,
                   "learning-rate": 0.0007 if pipeline == "event" else 0.0005,
                   "patience": 15 if pipeline == "event" else 5}
        if pipeline == "event":
            options.update({"finetune-all": True, "augment": True})
        if pipeline == "scratch":
            options["balanced-sampling"] = dataset != "pennfudan"
        if suffix == "person-baseline":
            options["baseline"] = True
        jobs.append({"version": f"{directory.name}-{suffix}", "dataset": dataset, "parent": parent,
                     "pipeline": pipeline, "options": options,
                     "dataset_sha256": digest(settings.data_dir / "datasets" / dataset / "manifest.json"),
                     "parent_sha256": digest(settings.data_dir / "runs" / parent / "weights.pt")})
    plan = {"name": directory.name, "selection": "validation only; calibration separate; test report only",
            "jobs": jobs, "unsupported": [
                "General 80-class/domain-inherited YOLO copies lack local labeled 80-class domain evaluation data.",
                "Fighting, theft, traffic collisions and industrial hazard events lack trained real-data candidates.",
                "SH17 has sparse PPE labels and classes absent from held-out splits; coverage is reported per class.",
                "These development sets do not establish independent camera/site acceptance or deployment readiness.",
                "Entity motion data are synthetic; improved synthetic accuracy does not validate real video events.",
                "Only CPU execution is available here; CUDA/FPGA performance is unverified.",
            ]}
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(create(parser.parse_args().directory))
