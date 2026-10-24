"""Explicit fine-tuning settings; no optimizer or training is executed here."""

import math


def training_policy(profile, continuing, learning_rate, epochs, batch, effective_batch_size=None, resume=False):
    if profile not in {"auto", "legacy-v1", "finetune-v2", "finetune-v3"}:
        raise ValueError("Unknown YOLO training profile")
    resolved = ("legacy-v1" if resume or (not continuing and learning_rate is None) else "finetune-v2") if profile == "auto" else profile
    if resume and resolved != "legacy-v1":
        raise ValueError("Changed fine-tuning policy requires a new version, not legacy optimizer resume")
    if effective_batch_size is not None and (isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int) or effective_batch_size < batch):
        raise ValueError("Effective batch must be at least the physical batch")
    rate = learning_rate if learning_rate is not None else (0.0001 if resolved in {"finetune-v2", "finetune-v3"} else None)
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("Learning rate must be finite and positive")
    arguments = {} if rate is None else {"optimizer": "AdamW", "lr0": rate, "cos_lr": True}
    if resolved in {"finetune-v2", "finetune-v3"}:
        arguments.update(warmup_bias_lr=0.0, warmup_epochs=min(1.0, epochs * 0.1),
            momentum=0.9, nbs=effective_batch_size or max(8, batch), scale=0.15, translate=0.05,
            hsv_h=0.01, hsv_s=0.3, hsv_v=0.2, iou=0.5, conf=0.001)
    elif effective_batch_size is not None:
        arguments["nbs"] = effective_batch_size
    if resolved == "finetune-v3":
        arguments.update(freeze=4, scale=0.25, translate=0.08, hsv_v=0.3, degrees=3.0,
                         mosaic=0.15, close_mosaic=min(5, max(1, epochs // 4)))
    return {"version": resolved, "requested": profile, "arguments": arguments,
        "effective_learning_rate": rate, "warmup_fraction_cap": 0.1 if resolved in {"finetune-v2", "finetune-v3"} else None,
        "lossless_prepared_images": resolved in {"finetune-v2", "finetune-v3"},
        "training_decode_mode": "bounded-jpeg-v1" if resolved in {"finetune-v2", "finetune-v3"} else "full-v1",
        "validation": "native AP50; RGB/geometry, single-label NMS .5, score floor .001, max100"
            if resolved in {"finetune-v2", "finetune-v3"} else "legacy Ultralytics fitness followed by native parent gate"}


def bounded_warmup_iterations(library_iterations, batches, epochs, fraction):
    if fraction is None:
        return library_iterations
    return min(library_iterations, max(0, int(batches * epochs * fraction)))
