"""Capability discovery never equates installed drivers with validated model execution."""

import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import psutil


ONNX_PROVIDER_ORDER = (
    "TensorrtExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
    "MIGraphXExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider",
    "CoreMLExecutionProvider", "XNNPACKExecutionProvider", "CPUExecutionProvider",
)


def recorded_benchmarks(data_dir):
    """Display bounded saved evidence separately from the active process runtime."""
    records = []
    for path in sorted((Path(data_dir) / "hardware").glob("*/report.json"))[-16:]:
        try:
            if path.stat().st_size > 1_000_000:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            records.append({"id": path.parent.name, "model": Path(report["model"]).name,
                            "scope": report["scope"], "runtime_version": report["runtime_version"],
                            "samples": report["samples"], "results": report["results"],
                            "live_activation": False})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def onnx_session_policy(available, requested=None, provider_options=None, threads=3):
    """Pure configuration policy; also used by hardware contract simulations."""
    requested = list(ONNX_PROVIDER_ORDER if requested is None else requested)
    selected = [name for name in requested if name in available]
    if not selected:
        raise RuntimeError("Requested accelerator execution provider unavailable")
    options = provider_options or {}
    if set(options) - set(requested):
        raise ValueError("Provider options must refer to requested providers")
    threads = max(1, min(int(threads), psutil.cpu_count(logical=False) or 1))
    configured = {name: dict(options.get(name, {})) for name in selected}
    if "XNNPACKExecutionProvider" in selected:
        configured["XNNPACKExecutionProvider"].setdefault("intra_op_num_threads", str(threads))
    return {
        "providers": selected,
        "provider_options": configured,
        "threads": 1 if "XNNPACKExecutionProvider" in selected else threads,
        "memory_pattern": "DmlExecutionProvider" not in selected,
        "serialize_runs": "DmlExecutionProvider" in selected,
        "allow_spinning": False,
    }


def torch_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def capabilities():
    provider_targets = {
        "TensorrtExecutionProvider": "tensorrt",
        "CUDAExecutionProvider": "cuda_onnx",
        "ROCMExecutionProvider": "rocm",
        "MIGraphXExecutionProvider": "migraphx",
        "OpenVINOExecutionProvider": "openvino",
        "DmlExecutionProvider": "directml",
        "CoreMLExecutionProvider": "coreml",
        "XNNPACKExecutionProvider": "xnnpack",
        "AzureExecutionProvider": "azure_onnx",
        "CPUExecutionProvider": "onnx_cpu",
    }
    result = {
        "training_device": "cpu",
        "onnx_providers": [],
        "torch": {"installed": False},
        "execution_targets": {
            "cpu": {"runtime_available": True, "project_status": "exercised"},
        },
        "fpga": {
            "runtime_installed": bool(importlib.util.find_spec("vart")),
            "status": "requires device-specific compiled artifact and validation",
        },
        "selection_policy": "Use an available native accelerator runtime; otherwise use calibrated CPU power profiles",
    }
    if importlib.util.find_spec("torch"):
        import torch

        device = torch_device()
        result["training_device"] = device
        result["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
        result["execution_targets"]["cuda_torch"] = {
            "runtime_available": torch.cuda.is_available(),
            "project_status": "available_unbenchmarked" if torch.cuda.is_available() else "unavailable",
        }
        result["execution_targets"]["mps"] = {
            "runtime_available": result["torch"]["mps_available"],
            "project_status": "available_unbenchmarked"
            if result["torch"]["mps_available"]
            else "unavailable",
        }
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        result["onnx_providers"] = ort.get_available_providers()
    available = set(result["onnx_providers"])
    for provider, name in provider_targets.items():
        installed = provider in available
        result["execution_targets"][name] = {
            "runtime_available": installed,
            "provider": provider,
            "project_status": "available_unbenchmarked"
            if installed and provider != "CPUExecutionProvider"
            else "exercised"
            if installed
            else "unavailable",
        }
    result["execution_targets"]["fpga_vitis"] = {
        "runtime_available": result["fpga"]["runtime_installed"],
        "project_status": "available_unbenchmarked"
        if result["fpga"]["runtime_installed"]
        else "unavailable",
    }
    return result


class ONNXExecutor:
    """CUDA/TensorRT/OpenVINO/DirectML/CoreML/Vitis via installed execution providers."""

    def __init__(self, path, providers=None, *, provider_options=None, threads=3,
                 strict=False, profile_prefix=None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        requested = list(ONNX_PROVIDER_ORDER if providers is None else providers)
        if strict and (not requested or requested[0] not in available):
            raise RuntimeError("Required primary execution provider unavailable")
        policy = onnx_session_policy(available, requested, provider_options, threads)
        self.policy = policy
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = policy["memory_pattern"]
        options.intra_op_num_threads = policy["threads"]
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        configured = [(name, policy["provider_options"][name]) for name in policy["providers"]]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=configured)
        self.providers = self.session.get_providers()
        if strict and requested[0] not in self.providers:
            raise RuntimeError("Runtime silently fell back from the required primary provider")
        self._lock = threading.Lock() if policy["serialize_runs"] else None

    def infer(self, named_inputs):
        start = time.perf_counter()
        with self._lock if self._lock is not None else nullcontext():
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
        if len(dpu) != 1 or len(children) != 1:
            raise ValueError("Adapter requires one DPU subgraph; mixed graphs need a board pipeline")
        self.runner = vart.Runner.create_runner(dpu[0], "run")

    def infer(self, inputs, outputs):
        job = self.runner.execute_async(inputs, outputs)
        status = self.runner.wait(job)
        if status != 0:
            raise RuntimeError(f"VART execution failed: {status}")
        return outputs
