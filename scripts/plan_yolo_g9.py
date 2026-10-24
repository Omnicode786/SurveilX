"""Freeze a YOLO-only, existing-data campaign. Does not execute training."""
import argparse
import json
from pathlib import Path
from training.datasets import digest
from scripts.train_accuracy_campaign import verify_job


def create(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Choose a new campaign directory")
    old = json.loads(Path("data/generations/accuracy-g8c-capacity-1015/plan.json").read_text())
    recipes = [
        ("fire-v3", "dfire-development-v2", "capacity-g8c-fire-yolo", 512, 4, .0003, "stretch", "finetune-v3", 48, 12),
        ("fire-control", "dfire-development-v2", "capacity-g8c-fire-yolo", 512, 4, .0002, "stretch", "finetune-v2", 48, 12),
        ("weapons-v3", "dangerous-items-development-v1", "capacity-g8c-weapons-yolo", 512, 4, .0002, "letterbox", "finetune-v3", 48, 12),
        ("weapons-control", "dangerous-items-development-v1", "capacity-g8c-weapons-yolo", 512, 4, .0003, "letterbox", "finetune-v2", 48, 12),
        ("ppe-v3", "sh17-development-v2", "capacity-g8c-ppe-yolo", 640, 2, .0002, "letterbox", "finetune-v3", 48, 12),
        ("ppe-control", "sh17-development-v2", "capacity-g8c-ppe-yolo", 640, 2, .0002, "letterbox", "finetune-v2", 48, 12),
        ("person-v3", "pennfudan", "accuracy-g2-person-yolo", 320, 8, .0001, "stretch", "finetune-v3", 16, 6),
    ]
    # Resolve the expanded D-Fire manifest name from the completed campaign.
    fire = next(j for j in old["jobs"] if j["version"] == "capacity-g8c-fire-yolo")
    code = {str(p).replace("\\", "/"): digest(p) for p in sorted(Path("training").glob("*.py"))}
    for name in ("scripts/train_accuracy_campaign.py", "surveilx/accelerators.py", "surveilx/training_runtime.py", "surveilx/config.py"):
        code[name] = digest(Path(name))
    jobs = []
    for suffix, dataset, parent, size, batch, rate, geometry, policy, epochs, patience in recipes:
        if suffix.startswith("fire"):
            dataset = fire["dataset"]
        manifest = Path("data/datasets") / dataset / "manifest.json"
        weights = Path("data/runs") / parent / "weights.pt"
        job = {"version": f"yolo-g9-{suffix}", "pipeline": "yolo", "dataset": dataset, "parent": parent,
            "dataset_sha256": digest(manifest), "parent_sha256": digest(weights), "code_sha256": code,
            "options": {"epochs": epochs, "patience": patience, "threads": 3, "seed": 42,
                "imgsz": size, "batch": batch, "learning-rate": rate, "preprocessing": geometry,
                "training-profile": policy, "effective-batch-size": 8},
            "reason": "New version initialized from preserved modified-YOLO weights; native validation selection, held-out calibration/test. No optimizer-state resume or automatic activation."}
        if (Path("data/runs") / job["version"]).exists():
            raise ValueError("Output version already exists")
        verify_job(job)
        jobs.append(job)
    plan = {"name": directory.name, "required_cuda": True, "gpu_memory_fraction": .9,
        "minimum_free_disk_bytes": 512*2**20, "minimum_available_ram_bytes": 1*2**30,
        "minimum_available_commit_bytes": 2*2**30, "minimum_free_gpu_mib": 500,
        "selection": "Validation only. Test is reporting only. No acceptance or deployment side effects.",
        "excluded": ["scratch/SVA", "temporal event training", "new datasets", "automatic production activation"],
        "limitations": ["Small development datasets cannot establish perfect accuracy or site independence.",
            "PPE rare classes require more real labels; YOLO cannot learn unannotated event classes.",
            "Seven serial experiments; resource reserve can stop the queue. Existing artifacts preserved."],
        "jobs": jobs}
    directory.mkdir(parents=True)
    path = directory / "plan.json"
    path.write_text(json.dumps(plan, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(create(parser.parse_args().directory))
