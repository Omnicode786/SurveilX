"""Bounded real scratch-model ONNX parity and provider-placement probe.

Uses validation images only. Does not register, calibrate or activate an export.
Optional isolated runtime path avoids changing the running training environment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from surveilx.accelerators import ONNXExecutor
from surveilx.hardware import probe
from training.detection_pipeline import DetectionImages, digest
from training.detector_model import SVADetector


class RawDetectorExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, context, zones):
        return tuple(value[key] for value in self.model(images, context, zones)
                     for key in ("box_raw", "objectness", "classes"))


def benchmark(run, dataset, output, provider="DmlExecutionProvider", device_ids=(0, 1),
              runtime_path=None, samples=3, repeats=5):
    if not 1 <= samples <= 16 or not 1 <= repeats <= 20:
        raise ValueError("Probe supports 1-16 samples and 1-20 repeats")
    if runtime_path:
        if "onnxruntime" in sys.modules:
            raise RuntimeError("Isolated runtime must be selected before importing ONNX Runtime")
        sys.path.insert(0, str(Path(runtime_path).resolve()))
    import onnxruntime as ort

    run, dataset, output = Path(run), Path(dataset), Path(output)
    metadata = json.loads((run / "manifest.json").read_text())
    data = json.loads(dataset.read_text())
    if metadata.get("architecture") != "sva-detector":
        raise ValueError("This probe supports SVA-Detector artifacts")
    if metadata["weights_sha256"] != digest(run / "weights.pt") or metadata["dataset_sha256"] != digest(dataset):
        raise ValueError("Frozen model/dataset checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = SVADetector(**metadata["model_config"]).eval()
    model.load_state_dict(torch.load(run / "weights.pt", map_location="cpu", weights_only=True))
    wrapper = RawDetectorExport(model).eval()
    size = metadata["config"]["image_size"]
    rows = [s for s in data["samples"] if s["split"] == "validation"][:samples]
    if not rows:
        raise ValueError("Validation samples required")
    loader = DetectionImages(dataset.parent, rows, image_size=size)
    inputs, expected = [], []
    for index in range(len(loader)):
        tensors = tuple(value[None] for value in loader[index][:3])
        inputs.append(dict(zip(("images", "context", "zones"), [v.numpy() for v in tensors])))
        with torch.no_grad():
            expected.append([v.numpy() for v in wrapper(*tensors)])
    graph = output / "detector.onnx"
    torch.onnx.export(wrapper, tuple(torch.from_numpy(v) for v in inputs[0].values()),
                      str(graph), input_names=list(inputs[0]), opset_version=18, dynamo=False)
    report = {"scope": "Real forward-output parity and latency on validation images; excludes decode/NMS and acceptance",
              "model": str(run), "weights_sha256": metadata["weights_sha256"],
              "onnx_sha256": digest(graph), "dataset_sha256": digest(dataset), "image_size": size,
              "samples": len(inputs), "repeats": repeats, "hardware": probe().json(),
              "runtime_version": ort.__version__, "runtime_path": ort.__file__,
              "available_providers": ort.get_available_providers(), "results": [],
              "deployment_eligible": False, "concurrent_cpu_training": True}
    configurations = [("CPUExecutionProvider", None)] + [(provider, i) for i in device_ids]
    for name, device in configurations:
        record = {"provider": name, "device_id": device, "passed": False}
        executor = None
        try:
            options = {} if device is None else {name: {"device_id": str(device)}}
            started = time.perf_counter()
            executor = ONNXExecutor(graph, [name] if device is None else [name, "CPUExecutionProvider"],
                                    provider_options=options, threads=1, strict=True,
                                    profile_prefix=output / f"profile-{name}-{device}")
            record["initialization_ms"] = (time.perf_counter() - started) * 1000
            times, error = [], 0.0
            for named, reference in zip(inputs, expected, strict=True):
                observed, _ = executor.infer(named)
                for actual, target in zip(observed, reference, strict=True):
                    error = max(error, float(np.max(np.abs(actual - target))))
                    np.testing.assert_allclose(actual, target, rtol=0.005, atol=0.001)
                for _ in range(repeats):
                    _, elapsed = executor.infer(named)
                    times.append(elapsed)
            record.update(passed=True, registered_providers=executor.providers,
                          max_absolute_error=error, p50_ms=float(np.median(times)),
                          p95_ms=float(np.percentile(times, 95)), measurements=len(times))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        finally:
            if executor is not None:
                trace = Path(executor.session.end_profiling())
                events = json.loads(trace.read_text())
                counts = Counter(e.get("args", {}).get("provider") for e in events
                                 if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
                record["node_execution_events"] = dict(counts)
                record["accelerator_nodes_observed"] = bool(counts.get(name)) if device is not None else False
        report["results"].append(record)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--provider", default="DmlExecutionProvider")
    parser.add_argument("--device-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--runtime-path")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    benchmark(**vars(parser.parse_args()))
