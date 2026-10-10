"""Bounded deterministic forward-only sensitivity check, not training or AP."""
import json
from collections import Counter
from pathlib import Path

import torch

from scripts.diagnose_sva_failures import attribute_failures
from training.detection_pipeline import DetectionImages, collate_detection, digest
from training.detector_model import SVADetector, decode_detections, flatten_predictions


def check():
    torch.set_num_threads(2)
    report = {"scope": "12 evenly spaced validation images per domain; forward only. Small descriptive sample, not AP or a causal augmentation ablation", "runs": []}
    for domain, dataset in (("fire", "dfire-development-v2"), ("weapons", "dangerous-items-development-v1"), ("ppe", "sh17-development-v2")):
        directory = Path("data/runs") / f"cuda-g4-{domain}-scratch"
        info = json.loads((directory / "manifest.json").read_text())
        manifest_path = Path("data/datasets") / dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert digest(manifest_path) == info["dataset_sha256"] and digest(directory / "weights.pt") == info["weights_sha256"]
        samples = [s for s in manifest["samples"] if s["split"] == "validation"][::8][:12]
        model = SVADetector(**info["model_config"]).eval()
        model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
        data = DetectionImages(manifest_path.parent, samples, image_size=info["config"]["image_size"])
        counts = {mode: Counter() for mode in ("original", "horizontal_flip", "brighten")}
        with torch.inference_mode():
            for item in data:
                images, context, zones, targets = collate_detection([item])
                for mode in counts:
                    x, z = images.clone(), zones.clone()
                    target = {k: v.clone() for k, v in targets[0].items() if k != "geometry"}
                    if mode == "horizontal_flip":
                        x, z = x.flip(-1), z.flip(-1)
                        target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]
                    elif mode == "brighten":
                        x = (x * 1.2 + .06).clamp(0, 1)
                    heads = model(x, context, z)
                    flat = flatten_predictions(heads)
                    probabilities, objectness = flat["classes"][0].sigmoid(), flat["objectness"][0, :, 0].sigmoid()
                    scores, labels = (probabilities * objectness[:, None]).max(-1)
                    raw = {"boxes": flat["boxes"][0], "scores": scores, "labels": labels,
                           "objectness": objectness, "class_probabilities": probabilities}
                    rows, _ = attribute_failures(raw, decode_detections(heads, score_threshold=.001)[0], target, info["calibration"])
                    counts[mode].update(row["category"] for row in rows)
        report["runs"].append({"domain": domain, "images": [s["image"] for s in samples], "counts": counts})
    Path("reports/g4-augmentation-sensitivity.json").write_text(json.dumps(report, indent=2))
    print("Forward sensitivity report saved")


if __name__ == "__main__":
    check()
