# Hardware implementation and validation

Verified 2026-10-03. Simulation, runtime installation, measured inference and live activation are separate states.

## This laptop

Windows registry and NVIDIA's own `nvidia-smi` identify a Dell Latitude 5591, Intel Core i7-8850H (six physical / twelve logical cores), 15.80 GiB RAM, Intel UHD Graphics 630 and NVIDIA GeForce MX130. The MX130 reports 2,048 MiB VRAM, compute capability 5.0 and driver 581.57. WMI/CIM inventory fails on this host; the product now uses a cached registry fallback for names.

The primary environment uses PyTorch 2.14.0+cpu and ONNX CPU/Azure. NVIDIA hardware is present even though this PyTorch installation cannot train on it. The [official PyTorch CUDA build guidance](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337) identifies CUDA 12.6 as the legacy build retaining Maxwell support. Do not blindly install a newer CUDA wheel. A separate compatible CUDA environment, a real kernel smoke test and a small-batch memory test are still required. CUDA installation was deferred with only about 3 GiB free and active training; no existing packages or training processes were replaced.

ONNX Runtime DirectML 1.24.4 was installed separately in `data/runtime/directml` with `--no-deps --no-cache-dir`. Probe subprocesses explicitly prepend this directory; the server and training environment remain separate. This provides real GPU inference, not CUDA training.

## Actual execution evidence

| Artifact | Size | ONNX CPU median forward | DirectML adapter 1 median forward | Evidence |
|---|---:|---:|---:|---|
| accuracy-g2-person-scratch | 256 | 112.0 ms | 19.7 ms | `data/hardware/directml-probe-g1/report.json` |
| accuracy-g3b-weapons-scratch | 384 | 234.9 ms | 40.0 ms | `data/hardware/directml-weapons-g1/report.json` |

Each probe used three validation images with five timed repetitions each, one CPU thread, full precision and fixed batch one. Profiling recorded actual DirectML node execution. Maximum raw-output error versus PyTorch was below 0.00004 across these probes. Adapter IDs are DXGI indices; do not assume they have the same numbering as another GPU inventory utility.

The person backend also completed the full 23-image validation split through `ONNXScratchDetector`: both backends AP50 0.409895, with mean forward/decode/NMS time 169.1 ms on PyTorch CPU and 41.1 ms on DirectML. This validates the detector adapter on that split. It does not change the historical untouched-test AP50, establish hazard accuracy, or approve live deployment. See `data/hardware/directml-probe-g1/validation.json`.

The weapons backend completed all 96 validation images: both backends AP50 0.077472, mean forward/decode/NMS 365.4 ms on PyTorch CPU and 62.2 ms on DirectML. The low score is preserved, not improved by acceleration. See `data/hardware/directml-weapons-g1/validation.json`.

Measurements ran alongside CPU training and include profiling overhead in the forward probes. They are preliminary host measurements, not dedicated-load throughput or power benchmarks. Initial graph compilation is excluded from warm timings (the first DirectML adapter initialization took about 104 seconds). No watts, energy savings or target-board throughput are inferred.

## Product changes

- The hardware screen shows the detected computer, runtime availability, recorded offline measurements and explicitly labeled simulation results.
- `ONNXExecutor` accepts per-provider options, bounded threads and profiling paths. Explicit strict mode rejects a missing primary provider or silent session fallback. Per-node profiling is still required to identify partial CPU placement.
- DirectML sessions disable memory patterns and serialize execution, following [ONNX Runtime's requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
- TensorRT precedes CUDA so unsupported TensorRT nodes can use CUDA before CPU, following [the provider guide](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).
- XNNPACK uses its own bounded thread pool, one ORT thread and disabled ORT spinning, following [the XNNPACK guidance](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html). Measure on the real target before tuning further.
- `ONNXScratchDetector` is an opt-in offline/acceptance adapter. It verifies source-weight and graph hashes, retains preprocessing, decoding and score calibration, and enforces a fixed batch-one image contract. Variable calibrated profiles and re-identification embeddings are not exported by this adapter. Exports stay outside frozen model directories and never auto-activate.
- VART rejects mixed CPU/DPU graphs; board-specific compiled artifacts and quantized inputs remain necessary. The [Vitis AI integration guide](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html) describes the separate vendor compilation/runtime workflow.

## Simulated contracts

`python -m scripts.simulate_hardware` writes `reports/hardware-simulation.json`. Eleven hypothetical configurations cover ordinary PCs, CUDA, TensorRT, Jetson, Raspberry Pi, low-memory edge devices, OpenVINO, DirectML, CoreML, ROCm and MIGraphX. They exercise the actual provider-order policy and governor under normal load, thermal pressure, low memory, low battery, CPU pressure and slow inference. They do not emulate accelerator kernels or predict accuracy, speed, energy or thermal behavior. VART graph/execution failure contracts are separately tested with doubles. Actual FPGA, Jetson, Pi, AMD and Apple execution remains unverified.

## Reproduce a new immutable probe

```powershell
.venv/Scripts/python.exe -m scripts.benchmark_accelerator data/runs/accuracy-g2-person-scratch data/datasets/pennfudan/manifest.json data/hardware/new-probe --runtime-path data/runtime/directml --device-ids 1
.venv/Scripts/python.exe -m scripts.validate_onnx_detector data/runs/accuracy-g2-person-scratch data/datasets/pennfudan/manifest.json data/hardware/new-probe data/runtime/directml --device-id 1
```

The benchmark requires a new output directory, verifies frozen model/data hashes and uses validation only. Deployment needs per-class evaluation, recalibration if precision changes, independent acceptance, measured target latency and normal canary/rollback gates. CUDA training and DirectML inference are distinct capabilities.

## Local CUDA training verified 2026-10-04

The MX130 is a Maxwell sm50 GPU with 2 GiB VRAM. An isolated `.venv-cuda` now runs PyTorch 2.14.0+cu126 and torchvision 0.29.0+cu126. The [official PyTorch compatibility notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) identifies CUDA 12.6 as the legacy build; automatic upgrades to a newer CUDA architecture build can remove Maxwell compatibility. Actual sm50 kernels and model optimization passed on this machine.

`data/hardware/cuda-training-g1/report.json` verifies frozen model/dataset hashes and uses real training samples only. Full precision, batch one, three CPU threads, two warmups and five measured forward/loss/backward/AdamW steps:

| Architecture | CPU median | CUDA median | Ratio | Peak PyTorch reserved |
|---|---:|---:|---:|---:|
| Scene event, 8 x 96 px | 48.82 ms | 15.20 ms | 3.21x | 42 MiB |
| Scratch detection, 512 px | 1029.05 ms | 182.21 ms | 5.65x | 236 MiB |
| Adapted YOLO, 512 px | 895.91 ms | 192.40 ms | 4.66x | 210 MiB |
| Entity event, 8 x 64 px | 26.42 ms | 20.33 ms | 1.30x | 26 MiB |

The entity probe is recorded separately in `data/hardware/cuda-training-entity-g1/report.json` and uses a generated training sample; it validates kernels rather than real-domain accuracy.

Gradients remained finite, optimizer steps changed the temporary model weights, and initial losses agreed between CPU and CUDA within 0.00006%. Changed probe weights were discarded. Reserved-memory figures exclude driver/context memory, and step timings exclude data loading, transfer preparation and full-epoch evaluation overhead. They do not predict power use or every future workload's speed.

The app uses `SURVEILX_TRAINING_PYTHON` for child training processes. Main inference remains in its original environment. Hardware status exposes matching recorded training evidence separately from the API process's runtime availability. Existing frozen campaigns retain their batch recipes.

### Measured batch tuning

`data/hardware/cuda-batch-tuning-g1/report.json` measures batches 1, 2, 4 and 8 using distinct real training samples, float32 loss/backward/AdamW steps, two warmups and three timed steps per trial. It chooses the highest measured throughput below a PyTorch reserved-memory limit of total VRAM minus 512 MiB. Driver/context and other applications consume additional memory; this is an offline selection rule, not an exclusive reservation.

| Architecture / contract | Selected batch | Samples/second | PyTorch reserved |
|---|---:|---:|---:|
| Adapted YOLO, 512 px | 8 | 10.92 | 1,336 MiB |
| Scratch detection, 512 px | 4 | 7.89 | 896 MiB |
| Scene event, 8 x 96 px | 8 | 166.01 | 180 MiB |

Scratch batch 8 reached 1,702 MiB and was rejected for headroom. The completed report records the laptop plugged in. Actual PPE YOLO batch-8 training has reached 99% GPU utilization with approximately 400-450 MiB free device memory. Utilization drops during loading, validation and checkpoint writes. Timing excludes these overheads; it is not a full-epoch throughput claim.

The laptop configuration now uses `SURVEILX_TRAINING_BATCH_SIZE=4`, `SURVEILX_TRAINING_YOLO_BATCH_SIZE=8` and `SURVEILX_TRAINING_EVENT_BATCH_SIZE=8`. Event batches scale down for larger frame-count/image-size contracts and never exceed the configured limit. These are measured settings for this laptop, not universal defaults. New immutable g4 campaigns run serially through CUDA; the dashboard exposes current GPU utilization and memory alongside architecture limits.

An actual complete GPU fighting continuation (`cuda-fighting-g2-scene`) trained eight epochs and then stopped by validation patience. It retained the parent checkpoint because validation loss did not improve; calibrated test accuracy and each class F1 remained 0.70. Its new artifact records device `cuda` and stays inactive/unaccepted. CUDA acceleration does not establish accuracy improvements or field reliability.

Reproduce a fresh immutable report:

```powershell
.venv-cuda/Scripts/python.exe -m scripts.benchmark_cuda_training data/hardware/cuda-training-new/report.json
```
