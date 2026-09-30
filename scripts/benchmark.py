import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np

from surveilx.controller import Candidate, Scheduler
from surveilx.hardware import probe
from surveilx.vision import Detection, HOGDetector, Tracker, generated_frame


def benchmark():
    frame = generated_frame(0, 1)
    detector, tracker, scheduler = HOGDetector(), Tracker(), Scheduler()
    samples = {"native_hog_ms": [], "python_tracker_ms": [], "python_scheduler_ms": []}
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(25):
        start = time.perf_counter()
        detector.infer(frame, 320)
        samples["native_hog_ms"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        detections = [
            Detection([i % 8 / 8, i // 8 / 4, i % 8 / 8 + 0.06, i // 8 / 4 + 0.12], "person", 0.9)
            for i in range(32)
        ]
        tracker.update(detections)
        samples["python_tracker_ms"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        scheduler.allocate([Candidate(str(i), 1, i + 1, 50) for i in range(4)], 200, 5)
        samples["python_scheduler_ms"].append((time.perf_counter() - start) * 1000)
    profiler.disable()
    output = Path("reports")
    output.mkdir(exist_ok=True)
    with (output / "native-profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
    result = {
        "scope": "HOG on generated image; tracker with 32 synthetic boxes; four-camera scheduler. Not accuracy or end-to-end performance",
        "hardware": probe().json(),
        "iterations": 25,
        "results": {
            key: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}
            for key, values in samples.items()
        },
        "decision": "OpenCV/PyTorch native kernels plus Python orchestration; no custom C++ without a measured bottleneck",
    }
    (output / "native-benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    benchmark()
