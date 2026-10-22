"""Warm runtime detector benchmarks on validation images, separate from cold evaluation."""

import gc
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from training.datasets import digest
from surveilx.detector_expert import ScratchDetector, AdaptedYOLODetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(2)
    datasets = {digest(p): p for p in Path("data/datasets").glob("*/manifest.json")}
    for report_path in sorted(Path("data/model-evaluations").glob("*.json")):
        if report_path.name == "status.json" or (args.versions and report_path.stem not in args.versions):
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("task") != "detection" or report.get("latency_usable_for_routing"):
            continue
        run = Path("data/runs") / report_path.stem
        metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if digest(run / "weights.pt") != report["weights_sha256"]:
            raise ValueError("Checkpoint changed")
        dataset = datasets[report["dataset_sha256"]]
        manifest = json.loads(dataset.read_text(encoding="utf-8"))
        paths = [dataset.parent / s["image"] for s in manifest["samples"] if s["split"] == "validation"][:8]
        images = [cv2.imread(str(p)) for p in paths]
        expert = (ScratchDetector if metadata["architecture"] == "sva-detector" else AdaptedYOLODetector)(run)
        elapsed = []
        for index, image in enumerate([*images[:2], *images]):
            if expert.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            expert.infer(image, None)
            if expert.device == "cuda":
                torch.cuda.synchronize()
            if index >= 2:
                elapsed.append((time.perf_counter() - start) * 1000)
        report.update(
            runtime_latency_ms=float(np.median(elapsed)),
            runtime_p95_ms=float(np.percentile(elapsed, 95)),
            latency_usable_for_routing=True,
            runtime_latency_scope="warm single-image runtime adapter, includes preprocessing, forward, decode and calibration; excludes file/camera I/O",
            runtime_latency_samples=len(elapsed),
            runtime_fps=1000 / float(np.mean(elapsed)),
            runtime_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20
            if expert.device == "cuda"
            else None,
            runtime_hardware=torch.cuda.get_device_name() if expert.device == "cuda" else "CPU",
            runtime_benchmark_sha256=digest(Path(__file__)),
        )
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp.replace(report_path)
        print(
            json.dumps({"version": report_path.stem, "median_ms": report["runtime_latency_ms"]}), flush=True
        )
        del expert
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
