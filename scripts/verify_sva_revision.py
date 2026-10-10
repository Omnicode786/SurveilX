"""Read-only labels/checkpoints verification. Never trains or saves weights."""
import json
from collections import Counter
from pathlib import Path

import torch

from scripts.analyze_sva_datasets import DATASETS, grid
from training.detector_model import SVADetector, assign_targets


def verify():
    torch.set_num_threads(2)
    report = {"scope": "Labels-only assignment and frozen-checkpoint compatibility; no training", "datasets": [], "checkpoints": []}
    for dataset, size in DATASETS:
        manifest = json.loads((Path("data/datasets") / dataset / "manifest.json").read_text())
        points, steps, levels = grid(size)
        flat = {"points": torch.from_numpy(points), "steps": torch.from_numpy(steps),
                "levels": torch.from_numpy(levels), "classes": torch.zeros(1, len(points), len(manifest["classes"])),
                "boxes": torch.zeros(1, len(points), 4)}
        totals, per_class = Counter(), {name: Counter() for name in manifest["classes"]}
        for sample in manifest["samples"]:
            if sample["split"] != "train":
                continue
            target = {"boxes": torch.tensor(sample["boxes"]).reshape(-1, 4), "labels": torch.tensor(sample["labels"], dtype=torch.long)}
            original = assign_targets(flat, [target], return_metadata=True)
            repaired = assign_targets(flat, [target], return_metadata=True, assignment="coverage-v2")
            assert torch.equal(original[0], repaired[0]), "Positive location set changed"
            old, new = set(original[3][0].tolist()), set(repaired[3][0].tolist())
            totals["images"] += 1
            for index, label in enumerate(sample["labels"]):
                values = {"objects": 1, "v1_unassigned": int(index not in old), "v2_unassigned": int(index not in new)}
                totals.update(values)
                per_class[manifest["classes"][label]].update(values)
        report["datasets"].append({"dataset": dataset, **totals, "per_class": per_class})
    all_runs = [p for p in Path("data/runs").glob("*/manifest.json")]
    for name in ("cuda-g4-fire-scratch", "cuda-g4-weapons-scratch", "cuda-g4-ppe-scratch"):
        run = Path("data/runs") / name
        info = json.loads((run / "manifest.json").read_text())
        state = torch.load(run / "weights.pt", map_location="cpu", weights_only=True)
        model = SVADetector(**info["model_config"]).eval()
        model.load_state_dict(state, strict=True)
        upgraded = SVADetector(**info["model_config"], architecture_version=2).eval()
        upgraded.load_v1_for_classification_upgrade(state)
        with torch.inference_mode():
            image = torch.zeros(1, 3, 64, 64)
            a, b = model(image), upgraded(image)
            identical = all(torch.equal(x[k], y[k]) for x, y in zip(a, b, strict=True) for k in x)
        assert identical
        parent_path = next(p for p in all_runs if json.loads(p.read_text()).get("weights_sha256") == info["config"]["initialization"])
        parent_state = torch.load(parent_path.parent / "weights.pt", map_location="cpu", weights_only=True)
        same = state.keys() == parent_state.keys() and all(torch.equal(state[k], parent_state[k]) for k in state)
        report["checkpoints"].append({"run": name, "parent": str(parent_path.parent), "parent_tensors_identical": same,
            "v1_strict_load": True, "v2_upgrade_forward_identical": identical,
            "v1_parameters": sum(p.numel() for p in model.parameters()), "v2_parameters": sum(p.numel() for p in upgraded.parameters()),
            "best_epoch": info["best_epoch"], "epochs_completed": info["epochs_completed"], "selection": info["validation_selection"]})
    output = Path("reports/g4-revision-verification.json")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
