import sys
from types import SimpleNamespace

import pytest

from surveilx.accelerators import ONNXExecutor, VitisExecutor, onnx_session_policy
from scripts.simulate_hardware import simulate


def test_hardware_api_distinguishes_simulation_and_saved_measurements(client):
    assert client.get("/api/hardware/status").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password-long"}).status_code == 200
    result = client.get("/api/hardware/status")
    assert result.status_code == 200
    payload = result.json()
    assert payload["simulation"]["hardware_executed"] is False
    assert len(payload["simulation"]["targets"]) == 11
    assert payload["recorded_benchmarks"] == []


def test_simulated_hardware_contracts_and_pressure():
    report = simulate()
    assert report["hardware_executed"] is False
    assert len(report["targets"]) == 11
    for row in report["targets"]:
        assert row["contract_checks_passed"]
        assert row["selection"]["providers"][-1] == "CPUExecutionProvider"


def test_onnx_order_thread_budget_and_directml_requirements():
    policy = onnx_session_policy(["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"])
    assert policy["providers"] == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    dml = onnx_session_policy(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert dml["serialize_runs"] and not dml["memory_pattern"]
    xnn = onnx_session_policy(["XNNPACKExecutionProvider", "CPUExecutionProvider"])
    assert xnn["threads"] == 1 and not xnn["allow_spinning"]
    assert int(xnn["provider_options"]["XNNPACKExecutionProvider"]["intra_op_num_threads"]) >= 1
    with pytest.raises(RuntimeError, match="unavailable"):
        onnx_session_policy(["CPUExecutionProvider"], ["CUDAExecutionProvider"])


def test_executor_applies_options_and_rejects_silent_fallback(monkeypatch):
    captured = {}

    class Options:
        def add_session_config_entry(self, key, value):
            captured[key] = value

    def session(path, sess_options, providers):
        captured.update(options=sess_options, providers=providers)
        return SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"], run=lambda *_: [])

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
        SessionOptions=Options, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=session))
    with pytest.raises(RuntimeError, match="silently fell back"):
        ONNXExecutor("model.onnx", ["DmlExecutionProvider"], strict=True,
                     provider_options={"DmlExecutionProvider": {"device_id": "1"}})
    assert captured["options"].enable_mem_pattern is False
    assert captured["options"].execution_mode == "sequential"
    assert captured["providers"][0][1]["device_id"] == "1"
    with pytest.raises(RuntimeError, match="unavailable"):
        ONNXExecutor("model.onnx", ["CUDAExecutionProvider"], strict=True)


def test_vitis_requires_single_dpu_and_checks_execution_status(monkeypatch):
    node = SimpleNamespace(has_attr=lambda _: True, get_attr=lambda _: "DPU")
    children = [node]
    runner = SimpleNamespace(execute_async=lambda *_: 3, wait=lambda _: 0)
    monkeypatch.setitem(sys.modules, "xir", SimpleNamespace(Graph=SimpleNamespace(
        deserialize=lambda _: SimpleNamespace(get_root_subgraph=lambda: SimpleNamespace(
            toposort_child_subgraph=lambda: children)))))
    monkeypatch.setitem(sys.modules, "vart", SimpleNamespace(Runner=SimpleNamespace(create_runner=lambda *_: runner)))
    executor = VitisExecutor("simulated.xmodel")
    assert executor.infer([1], [2]) == [2]
    runner.wait = lambda _: -1
    with pytest.raises(RuntimeError, match="VART execution failed"):
        executor.infer([1], [2])
    children.append(node)
    with pytest.raises(ValueError, match="one DPU"):
        VitisExecutor("simulated.xmodel")
    children[1] = SimpleNamespace(has_attr=lambda _: True, get_attr=lambda _: "CPU")
    with pytest.raises(ValueError, match="mixed graphs"):
        VitisExecutor("simulated.xmodel")


def test_real_onnx_scratch_adapter_parity_and_integrity(tmp_path):
    import json
    import numpy as np
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from scripts.benchmark_accelerator import RawDetectorExport
    from surveilx.onnx_detector import ONNXScratchDetector
    from training.detector_model import SVADetector, decode_detections
    from training.detection_pipeline import digest

    torch.set_num_threads(1)
    model = SVADetector(classes=3, width=8, depth=1).eval()
    inputs = (torch.rand(1, 3, 64, 64), torch.zeros(1, 4), torch.zeros(1, 1, 64, 64))
    run, exported = tmp_path / "run", tmp_path / "export"
    run.mkdir()
    exported.mkdir()
    torch.save(model.state_dict(), run / "weights.pt")
    graph = exported / "detector.onnx"
    torch.onnx.export(RawDetectorExport(model), inputs, str(graph), opset_version=18,
                      input_names=["images", "context", "zones"], dynamo=False)
    sha = digest(run / "weights.pt")
    (run / "manifest.json").write_text(json.dumps({"task": "detection", "architecture": "sva-detector",
        "weights_sha256": sha, "classes": ["a", "b", "c"], "config": {"image_size": 64}}))
    (exported / "report.json").write_text(json.dumps({"weights_sha256": sha,
        "onnx_sha256": digest(graph), "image_size": 64}))
    expert = ONNXScratchDetector(run, exported, ["CPUExecutionProvider"])
    with torch.no_grad():
        expected = decode_detections(model(*inputs), score_threshold=0.001)[0]
        actual = decode_detections(expert.model(*inputs), score_threshold=0.001)[0]
    for key in ("boxes", "scores", "labels"):
        np.testing.assert_allclose(actual[key].numpy(), expected[key].numpy(), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="fixed batch-one"):
        expert.model(torch.zeros(2, 3, 64, 64))
    with graph.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="frozen model"):
        ONNXScratchDetector(run, exported, ["CPUExecutionProvider"])
