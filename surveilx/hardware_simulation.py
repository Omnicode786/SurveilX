"""Exercise production selection/governor contracts with hypothetical hardware.

No timings, device execution, energy consumption or model accuracy are simulated.
"""

from dataclasses import replace

from surveilx.accelerators import onnx_session_policy
from surveilx.hardware import Hardware, PowerGovernor


TARGETS = {
    "ordinary_pc": (6, 16, ["CPUExecutionProvider"]),
    "cuda_workstation": (12, 32, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    "tensorrt_workstation": (12, 32, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
    "jetson": (6, 8, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
    "raspberry_pi": (4, 4, ["XNNPACKExecutionProvider", "CPUExecutionProvider"]),
    "low_memory_edge": (4, 2, ["CPUExecutionProvider"]),
    "intel_openvino": (8, 16, ["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
    "windows_directml": (6, 16, ["DmlExecutionProvider", "CPUExecutionProvider"]),
    "apple_coreml": (8, 16, ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    "amd_rocm": (8, 16, ["ROCMExecutionProvider", "CPUExecutionProvider"]),
    "amd_migraphx": (8, 16, ["MIGraphXExecutionProvider", "CPUExecutionProvider"]),
}


def simulate():
    rows = []
    for name, (cores, ram, providers) in TARGETS.items():
        hardware = Hardware("simulated", "simulated", cores, ram, ram / 2, 20, 50,
                            None, 45, 100, True)
        governor = PowerGovernor()
        levels = [governor.update(hardware)["level"] for _ in range(20)]
        healthy = levels[-1]
        stress = {}
        for label, changes in {
            "thermal": {"temperature_c": 90}, "memory": {"available_gb": 0.2},
            "battery": {"plugged_in": False, "battery_percent": 10},
            "cpu": {"cpu_percent": 99},
        }.items():
            governor = PowerGovernor()
            for _ in range(20):
                governor.update(hardware)
            stress[label] = [governor.update(replace(hardware, **changes))["level"] for _ in range(2)]
            assert stress[label][-1] == "economy", (name, label)
        assert all(level == "economy" for level in levels[:9])
        governor = PowerGovernor()
        for _ in range(20):
            governor.update(hardware)
        assert governor.update(hardware, latency_ms=900)["level"] != "performance"
        rows.append({"target": name, "hypothetical_cores": cores, "hypothetical_ram_gb": ram,
                     "selection": onnx_session_policy(providers), "healthy_level": healthy,
                     "stress_levels": stress, "contract_checks_passed": True})
    return {"kind": "software_contract_simulation", "hardware_executed": False,
            "scope": "Hypothetical provider availability and telemetry; no speed/accuracy/energy predictions",
            "targets": rows,
            "fpga": "VART graph and execution contracts tested with doubles in tests/test_accelerators.py; board validation pending"}


