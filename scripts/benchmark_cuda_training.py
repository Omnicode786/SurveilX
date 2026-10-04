"""Exercise real artifact loss/backward/optimizer steps on CPU and the laptop GPU.

Uses training samples only. Does not save changed weights or claim accuracy gains.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F

from training.datasets import digest
from training.detection_pipeline import DetectionImages, collate_detection
from training.detector_model import SVADetector, detection_loss
from training.models import SVANet, SVASceneNet
from training.pipeline import Clips


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(kind, device, batch_size=1):
    scene = kind == "scene"
    event = kind in {"scene", "entity"}
    version = {
        "scene": "fighting-g1-scene",
        "entity": "accuracy-g2-entity-v1",
        "scratch": "accuracy-ppe-g3b-scratch",
        "yolo": "accuracy-ppe-g3-yolo",
    }[kind]
    artifact = Path("data/runs") / version
    metadata = read(artifact / "manifest.json")
    if digest(artifact / "weights.pt") != metadata["weights_sha256"]:
        raise ValueError("Benchmark artifact checksum mismatch")
    dataset = Path("data/datasets") / (
        "airtlab-fighting-development-v1" if scene else "generated-v1" if event else "sh17-development-v2"
    )
    manifest = read(dataset / "manifest.json")
    if digest(dataset / "manifest.json") != metadata["dataset_sha256"]:
        raise ValueError("Benchmark dataset checksum mismatch")
    samples = [s for s in manifest["samples"] if s["split"] == "train" and (event or s["labels"])]
    if event:
        model = (SVASceneNet if scene else SVANet)(
            classes=len(metadata["classes"]), **metadata["architecture"]
        )
        model.load_state_dict(torch.load(artifact / "weights.pt", map_location="cpu", weights_only=True))
        values = [Clips(dataset, samples, scene=scene)[i] for i in range(batch_size)]
        inputs = [torch.stack([value[j] for value in values]).to(device) for j in range(3)]
        target = torch.tensor([value[3] for value in values], device=device)

        def loss():
            return F.cross_entropy(model(*inputs)["event"], target)

        size = metadata["input_contract"]["image_size"]
    else:
        size = 512
        values = DetectionImages(dataset, samples, size)
        images, context, zones, targets = collate_detection([values[i] for i in range(batch_size)])
        images, context, zones = [v.to(device) for v in (images, context, zones)]
        if kind == "scratch":
            model = SVADetector(**metadata["model_config"])
            model.load_state_dict(torch.load(artifact / "weights.pt", map_location="cpu", weights_only=True))

            def loss():
                return detection_loss(
                    model(images, context, zones),
                    targets,
                    classification_weight=metadata["config"]["classification_weight"],
                )["loss"]
        else:
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
            from training.yolo_model import ConditionedBlock

            model = YOLO(str(artifact / "weights.pt")).model.float()
            if not any(isinstance(block, ConditionedBlock) for block in model.modules()):
                raise ValueError("Expected the product's adapted YOLO architecture")
            model.args = get_cfg(overrides=model.args if isinstance(model.args, dict) else vars(model.args))
            boxes = torch.cat([target["boxes"] for target in targets]).to(device)
            batch = {
                "img": images,
                "batch_idx": torch.cat(
                    [
                        torch.full((len(target["boxes"]),), i, device=device)
                        for i, target in enumerate(targets)
                    ]
                ),
                "cls": torch.cat([target["labels"] for target in targets]).float().reshape(-1, 1).to(device),
                "bboxes": torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], -1),
            }

            def loss():
                return model(batch)[0].sum()

    model = model.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return (
        model,
        loss,
        {
            "model": version,
            "weights_sha256": metadata["weights_sha256"],
            "dataset_sha256": metadata["dataset_sha256"],
            "sample_split": "train",
            "image_size": size,
            "batch_size": batch_size,
        },
    )


def measure(kind, device, steps, batch_size=1):
    torch.manual_seed(42)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, loss_function, identity = build(kind, device, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    timings, losses = [], []
    for index in range(steps + 2):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if kind in {"scene", "entity"} else 5.0)
        if index == 0:
            gradients = [p for p in model.parameters() if p.grad is not None]
            if not gradients or not torch.stack([torch.isfinite(p.grad).all() for p in gradients]).all():
                raise RuntimeError("Missing or non-finite training gradients")
            nonzero = torch.stack([p.grad.abs().max() > 0 for p in gradients]).cpu().tolist()
            changed = next(
                (p for p, nonzero_grad in zip(gradients, nonzero, strict=True) if nonzero_grad), None
            )
            if changed is None:
                raise RuntimeError("All training gradients are zero")
            before = changed.detach().clone()
        optimizer.step()
        if index == 0 and torch.equal(before, changed):
            raise RuntimeError("Optimizer did not change the weights")
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if index >= 2:
            timings.append(elapsed)
        losses.append(float(loss.detach().cpu()))
    if any(p.device.type != device for p in model.parameters()):
        raise RuntimeError("Model did not execute on the requested device")
    result = {
        **identity,
        "device": device,
        "steps": steps,
        "warmup_steps": 2,
        "median_step_ms": statistics.median(timings),
        "step_ms": timings,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "optimizer_changed_weights": True,
        "samples_per_second": batch_size * 1000 / statistics.median(timings),
    }
    if device == "cuda":
        result.update(
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        )
    del model, loss_function, optimizer, gradients, before, changed, loss
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run(output, steps=5, architectures=("scene", "scratch", "yolo")):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new report path; recorded evidence is immutable")
    if not torch.cuda.is_available() or "sm_50" not in torch.cuda.get_arch_list():
        raise RuntimeError("The installed CUDA runtime must support this Maxwell GPU")
    torch.set_num_threads(3)
    properties = torch.cuda.get_device_properties(0)
    report = {
        "scope": "Real training-sample loss/backward/AdamW kernel and memory checks; excludes data loading, full epochs and accuracy evaluation",
        "created": time.time(),
        "python": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_mib": properties.total_memory / 1024**2,
        "precision": "float32",
        "cpu_threads": 3,
        "results": [],
        "state": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for kind in architectures:
            cpu, cuda = measure(kind, "cpu", steps), measure(kind, "cuda", steps)
            relative_loss_error = abs(cuda["initial_loss"] - cpu["initial_loss"]) / max(
                abs(cpu["initial_loss"]), 1e-8
            )
            if relative_loss_error > 0.005:
                raise RuntimeError(f"{kind}: CPU/GPU initial training loss differs by more than 0.5 percent")
            report["results"].append(
                {
                    "architecture": kind,
                    "cpu": cpu,
                    "cuda": cuda,
                    "speedup": cpu["median_step_ms"] / cuda["median_step_ms"],
                    "initial_loss_relative_error": relative_loss_error,
                }
            )
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report["results"][-1]), flush=True)
        report["state"] = "completed"
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("scene", "entity", "scratch", "yolo"),
        default=["scene", "scratch", "yolo"],
    )
    args = parser.parse_args()
    if not 3 <= args.steps <= 20:
        parser.error("steps must be 3..20")
    run(args.output, args.steps, args.architectures)
