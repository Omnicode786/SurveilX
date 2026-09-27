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
from training.models import SVANet


class Clips(Dataset):
    def __init__(self, root, samples):
        self.root, self.samples = root, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with np.load(self.root / item["file"], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["clip"].astype("float32")),
                torch.from_numpy(data["boxes"].astype("float32")),
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
    model = SVANet(classes=len(manifest["classes"]), **flags).to(device)
    if initialize_from:
        model.load_state_dict(torch.load(initialize_from, map_location=device, weights_only=True))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("context.", "event_head.")))
    loaders = {
        split: DataLoader(
            Clips(manifest_path.parent, [s for s in manifest["samples"] if s["split"] == split]),
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
        "name": "sva-net-entity-prototype",
        "stage": "candidate",
        "synthetic": manifest.get("synthetic", False),
        "domain": manifest["domain"],
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
