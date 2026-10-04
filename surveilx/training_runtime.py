"""Optional isolated training interpreter; inference keeps its own environment."""

from pathlib import Path
import json
import sys

from surveilx.config import settings


def training_interpreter():
    if not settings.training_python:
        return sys.executable
    path = Path(settings.training_python).resolve()
    if not path.is_file():
        raise ValueError("Configured training Python is missing; restore the verified environment")
    return str(path)


def training_batch_arguments(module="training.detection_pipeline", manifest=None):
    specific = (
        settings.training_yolo_batch_size
        if module == "training.yolo_pipeline"
        else settings.training_event_batch_size
        if module == "training.pipeline"
        else None
    )
    value = settings.training_batch_size if specific is None else specific
    if value is None:
        return []
    if isinstance(value, bool) or not 1 <= value <= 256:
        raise ValueError("Configured training batch size must be 1..256")
    if module == "training.pipeline" and manifest:
        # Scale down for larger user clip contracts; never exceed the configured limit.
        contract = manifest.get("input_contract", {})
        frames, size = contract.get("frames", 8), contract.get("image_size", 64)
        value = max(1, min(value, int(value * 8 * 96**2 / (frames * size**2))))
    return ["--batch" if module == "training.yolo_pipeline" else "--batch-size", str(value)]


def training_runtime_status():
    try:
        interpreter, error = training_interpreter(), None
    except ValueError as exc:
        interpreter, error = settings.training_python, str(exc)
    verification = None
    if settings.training_python and error is None:
        for path in reversed(sorted((settings.data_dir / "hardware").glob("cuda-training-*/report.json"))):
            try:
                if path.stat().st_size > 1_000_000:
                    continue
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("state") == "completed" and Path(report["python"]).resolve() == Path(
                    interpreter
                ):
                    verification = {
                        key: report[key]
                        for key in (
                            "scope",
                            "created",
                            "torch",
                            "cuda_runtime",
                            "gpu",
                            "precision",
                            "results",
                        )
                    }
                    break
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return {
        "configured": bool(settings.training_python),
        "python": interpreter,
        "error": error,
        "batch_size": settings.training_batch_size,
        "architecture_batch_limits": {
            "scratch": settings.training_batch_size,
            "yolo": settings.training_yolo_batch_size,
            "event": settings.training_event_batch_size,
        },
        "recorded_validation": verification,
        "scope": "Training subprocess configuration; each completed model records its actual device",
    }
