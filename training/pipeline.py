import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from surveilx.accelerators import torch_device
from training.calibration import fit_temperature, metrics
from training.datasets import digest, generate, validate_manifest
from training.models import SVANet, SVASceneNet


class Clips(Dataset):
    def __init__(self, root, samples, scene=False):
        self.root, self.samples, self.scene = root, samples, scene

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with np.load(self.root / item["file"], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["clip"].astype("float32")),
                torch.empty((len(data["clip"]), 0, 4))
                if self.scene
                else torch.from_numpy(data["boxes"].astype("float32")),
                torch.from_numpy(data["context"].astype("float32")),
                item["label"],
            )


def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for clip, boxes, context, label in loader:
            logits.append(model(clip.to(device), boxes.to(device), context.to(device))["event"].cpu().numpy())
            labels.extend(label.numpy().tolist())
    return np.concatenate(logits), np.asarray(labels)


def train(manifest_path, output, epochs=5, seed=42, ablation="full", initialize_from=None):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, counts = validate_manifest(manifest_path)
    if (output / "manifest.json").exists():
        raise ValueError("Run already exists; choose a new output version")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch_device()
    flags = {
        "use_motion": ablation != "no_motion",
        "use_context": ablation != "no_context",
        "use_interactions": ablation != "no_interactions",
        "use_temporal": ablation != "no_temporal",
    }
    scene = manifest.get("representation") == "scene_clip"
    if scene:
        if ablation == "no_interactions":
            raise ValueError("Scene clips have no entity interaction head to ablate")
        flags["use_interactions"] = False
    model = (SVASceneNet if scene else SVANet)(classes=len(manifest["classes"]), **flags).to(device)
    initialization = "random"
    if initialize_from:
        previous = Path(initialize_from).resolve()
        previous_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        if (
            previous_manifest.get("task") != "event"
            or previous_manifest.get("classes") != manifest["classes"]
            or previous_manifest.get("representation", "entity_clip")
            != manifest.get("representation", "entity_clip")
            or previous_manifest.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
            != manifest.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
            or previous_manifest.get("architecture") != flags
        ):
            raise ValueError("Continuation requires identical event classes, representation, contract and architecture")
        if digest(previous / "weights.pt") != previous_manifest.get("weights_sha256"):
            raise ValueError("Continuation checkpoint checksum mismatch")
        model.load_state_dict(torch.load(previous / "weights.pt", map_location=device, weights_only=True))
        initialization = previous_manifest["weights_sha256"]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("context.", "event_head.")))
    loaders = {
        split: DataLoader(
            Clips(manifest_path.parent, [s for s in manifest["samples"] if s["split"] == split], scene=scene),
            batch_size=8,
            shuffle=split == "train",
            num_workers=0,
        )
        for split in counts
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
    started = time.perf_counter()
    best, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        losses = []
        for clip, boxes, context, label in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            result = model(clip.to(device), boxes.to(device), context.to(device))
            # Only labeled event loss; unlabeled entity/relation heads are not falsely supervised.
            loss = F.cross_entropy(result["event"], label.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        validation_logits, validation_labels = predict(model, loaders["validation"], device)
        val_loss = F.cross_entropy(torch.tensor(validation_logits), torch.tensor(validation_labels)).item()
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss}
        )
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), output / "weights.pt")
    model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
    calibration_logits, calibration_labels = predict(model, loaders["calibration"], device)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    test_logits, test_labels = predict(model, loaders["test"], device)
    report = metrics(test_logits, test_labels, temperature)
    before = metrics(test_logits, test_labels)
    np.savez(output / "heldout_predictions.npz", logits=test_logits, labels=test_labels)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    artifact = {
        "schema_version": 1,
        "task": "event",
        "input_contract": manifest.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64}),
        "name": "sva-scene-clip-prototype" if scene else "sva-net-entity-prototype",
        "representation": "scene_clip" if scene else "entity_clip",
        "stage": "candidate",
        "synthetic": manifest.get("synthetic", False),
        "domain": manifest["domain"],
        "capability_ids": manifest.get("capability_ids", []),
        "classes": manifest["classes"],
        "dataset_sha256": digest(manifest_path),
        "weights_sha256": digest(output / "weights.pt"),
        "dataset_counts": counts,
        "temperature": temperature,
        "calibration_task": "event",
        "calibrated": True,
        "metrics": report,
        "uncalibrated_metrics": before,
        "history": history,
        "seed": seed,
        "epochs": epochs,
        "ablation": ablation,
        "architecture": flags,
        "initialization": initialization,
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "code_commit": commit,
        "duration_seconds": time.perf_counter() - started,
        "trainable_heads": ["event"],
        "trained_entity_states": False,
        "trained_relations": False,
        "deployment_eligible": False,
        "gate_reason": "Requires independent domain acceptance and reviewed deployment; synthetic runs cannot validate surveillance",
    }
    (output / "manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(
        description="Generate or import, train, evaluate and calibrate a versioned candidate"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    generated = sub.add_parser("generate")
    generated.add_argument("directory")
    generated.add_argument("--count", type=int, default=160)
    generated.add_argument("--seed", type=int, default=42)
    training = sub.add_parser("train")
    training.add_argument("manifest")
    training.add_argument("output")
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--ablation",
        choices=["full", "no_motion", "no_context", "no_interactions", "no_temporal"],
        default="full",
    )
    training.add_argument("--initialize-from")
    args = parser.parse_args()
    if args.command == "generate":
        print(generate(args.directory, args.count, args.seed))
    else:
        print(
            json.dumps(
                train(
                    args.manifest, args.output, args.epochs, args.seed, args.ablation, args.initialize_from
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
