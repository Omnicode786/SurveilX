"""Atomic optimizer checkpoints with explicit integrity and recipe binding."""

import random
from pathlib import Path
import numpy as np
import torch
from training.datasets import digest


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_training_state(path, model, optimizer, scheduler, epoch, recipe, best_state, history, sampler=None):
    atomic_save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "recipe": recipe,
            "best_state": best_state,
            "history": history,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler_rng": sampler.generator.get_state() if sampler is not None else None,
        },
        path,
    )


def load_training_state(path, expected_sha256, recipe):
    if not expected_sha256 or digest(path) != expected_sha256:
        raise ValueError("Resume checkpoint checksum mismatch")
    # Local, checksum-pinned optimizer/RNG checkpoint; never downloaded untrusted pickle.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved.get("recipe") != recipe:
        raise ValueError("Resume checkpoint recipe/model/data differs")
    return saved


def restore_rng(saved, sampler=None):
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if sampler is not None and saved["sampler_rng"] is not None:
        sampler.generator.set_state(saved["sampler_rng"])


def step_accumulated(optimizer, model, microbatches, maximum_norm=5.0):
    """Mean of microbatch objectives; handles a short final group without underweighting."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(microbatches)
    torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
