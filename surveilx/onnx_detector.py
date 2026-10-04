"""Opt-in, fixed-contract ONNX scratch backend for offline acceptance.

Export reports remain outside frozen model directories. Loading an export does
not register or activate a model; existing deployment acceptance still applies.
"""

import json
from pathlib import Path

import torch

from surveilx.accelerators import ONNXExecutor
from surveilx.detector_expert import ScratchDetector
from training.detection_pipeline import digest


class ONNXRawScratch(torch.nn.Module):
    def __init__(self, graph, image_size, providers, provider_options=None):
        super().__init__()
        self.image_size = image_size
        self.executor = ONNXExecutor(graph, providers, provider_options=provider_options, strict=True)

    def forward(self, images, context=None, zones=None):
        size = self.image_size
        if tuple(images.shape) != (1, 3, size, size):
            raise ValueError("ONNX scratch export requires its fixed batch-one image contract")
        if context is None:
            context = images.new_zeros((1, 4))
        if zones is None:
            zones = images.new_zeros((1, 1, size, size))
        if tuple(context.shape) != (1, 4) or tuple(zones.shape) != (1, 1, size, size):
            raise ValueError("ONNX scratch export context/zone contract mismatch")
        values, _ = self.executor.infer({name: tensor.detach().cpu().numpy()
                                       for name, tensor in zip(("images", "context", "zones"),
                                                               (images, context, zones), strict=True)})
        if len(values) != 9:
            raise ValueError("Expected three raw detection tensors per pyramid level")
        heads = []
        for offset in range(0, 9, 3):
            head = {key: torch.from_numpy(value) for key, value in zip(
                ("box_raw", "objectness", "classes"), values[offset:offset + 3], strict=True)}
            # Detection-only backend: decoder requires an embedding tensor but
            # Detection objects never expose it; this is not a re-ID export.
            head["embeddings"] = torch.zeros_like(head["objectness"])
            heads.append(head)
        return heads


class ONNXScratchDetector(ScratchDetector):
    def __init__(self, directory, export_directory, providers, provider_options=None):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        export_directory = Path(export_directory).resolve()
        report = json.loads((export_directory / "report.json").read_text())
        graph = export_directory / "detector.onnx"
        if (self.manifest.get("architecture") != "sva-detector"
                or self.manifest.get("task") != "detection"
                or self.manifest.get("inference_profiles")
                or report.get("weights_sha256") != self.manifest.get("weights_sha256")
                or digest(self.directory / "weights.pt") != report.get("weights_sha256")
                or digest(graph) != report.get("onnx_sha256")
                or report.get("image_size") != self.manifest["config"]["image_size"]):
            raise ValueError("ONNX export does not match the frozen model and fixed image contract")
        self.device = "cpu"  # Host preprocessing/decoding; executor owns accelerator tensors.
        self.model = ONNXRawScratch(graph, report["image_size"], providers, provider_options)
        self.name, self.version = "sva-detector-onnx", self.directory.name
        self.calibration = self.manifest.get("calibration", {})
        self.calibrated = self.calibration.get("status") == "fitted"
        self.classes = self.manifest["classes"]
