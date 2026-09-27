"""Capability discovery never equates installed drivers with validated model execution."""

import importlib.util
import time


def torch_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def capabilities():
    result = {
        "training_device": "cpu",
        "onnx_providers": [],
        "fpga": {
            "runtime_installed": bool(importlib.util.find_spec("vart")),
            "status": "requires device-specific compiled artifact and validation",
        },
    }
    if importlib.util.find_spec("torch"):
        result["training_device"] = torch_device()
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        result["onnx_providers"] = ort.get_available_providers()
    return result


class ONNXExecutor:
    """CUDA/TensorRT/OpenVINO/DirectML/CoreML/Vitis via installed execution providers."""

    def __init__(self, path, providers=None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        requested = providers or [
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [provider for provider in requested if provider in available]
        if not selected:
            raise RuntimeError("Requested accelerator execution provider unavailable")
        self.session = ort.InferenceSession(str(path), providers=selected)
        self.providers = self.session.get_providers()

    def infer(self, named_inputs):
        start = time.perf_counter()
        outputs = self.session.run(None, named_inputs)
        return outputs, (time.perf_counter() - start) * 1000


class VitisExecutor:
    """Actual VART execution interface for a user-supplied compiled DPU subgraph.

    Caller supplies board-specific quantized arrays matching tensor metadata.
    No unsupported promise of universal FPGA graph compilation or FPGA training.
    """

    def __init__(self, xmodel):
        import vart
        import xir

        self.graph = xir.Graph.deserialize(str(xmodel))
        children = self.graph.get_root_subgraph().toposort_child_subgraph()
        dpu = [node for node in children if node.has_attr("device") and node.get_attr("device") == "DPU"]
        if len(dpu) != 1:
            raise ValueError("Adapter requires one DPU subgraph; mixed graphs need a board pipeline")
        self.runner = vart.Runner.create_runner(dpu[0], "run")

    def infer(self, inputs, outputs):
        job = self.runner.execute_async(inputs, outputs)
        status = self.runner.wait(job)
        if status != 0:
            raise RuntimeError(f"VART execution failed: {status}")
        return outputs
