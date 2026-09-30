"""Compare the fallback with the available native C++/accelerator suppression op."""

import json
import time
from pathlib import Path

import torch
from training.detector_model import native_nms, python_nms


def benchmark():
    if native_nms is None:
        raise RuntimeError("Install a compatible TorchVision build for this comparison")
    torch.set_num_threads(3)
    torch.manual_seed(42)
    xy = torch.rand(600, 2) * 0.8
    boxes = torch.cat([xy, xy + torch.rand(600, 2) * 0.2], 1)
    scores = torch.rand(600)
    assert torch.equal(python_nms(boxes, scores), native_nms(boxes, scores, 0.5)[:100])
    timings = {}
    for name, function in (
        ("python", lambda: python_nms(boxes, scores)),
        ("native", lambda: native_nms(boxes, scores, 0.5)[:100]),
    ):
        start = time.perf_counter()
        for _ in range(25):
            function()
        timings[name] = (time.perf_counter() - start) * 1000 / 25
    report = {
        "mean_ms": timings,
        "same_retained_indices": True,
        "boxes": 600,
        "maximum_retained": 100,
        "iterations": 25,
        "device": "cpu",
        "torch": torch.__version__,
        "scope": "NMS microbenchmark only; not total detector acceleration",
    }
    Path("reports/nms-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
