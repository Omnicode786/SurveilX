# Deployment and acceleration

## Normal computer

Windows, Linux and macOS use the same Python backend and built React frontend. CPU-only operation needs no CUDA. HOG provides a real person-detection baseline without downloaded weights; generated streams use a separately named color fixture. SQLite plus encrypted filesystem objects is the lightweight development profile. PostgreSQL is the deployment database.

The governor probes CPU, RAM, optional NVIDIA telemetry, temperature and battery where available. It starts at economy, promotes after ten healthy epochs, and reduces workload immediately under configured pressure. Levels select resolution, capture rate and inference-time budget. Unknown temperature or watts stays unknown. It does not change OS power plans. Initial limits are engineering defaults, not learned optimums.

## GPU / CUDA

Current laptop inventory, actual DirectML detector measurements, compatibility findings and executable simulations are recorded in [hardware-validation.md](hardware-validation.md). The laptop has an MX130; its main PyTorch environment is CPU-only. An isolated DirectML runtime has now executed person and weapons scratch exports, while CUDA training and live ONNX activation remain pending.

Install a compatible official CUDA-enabled PyTorch build for the driver and target platform. The current installed runtime is discovered at execution; CPU-only PyTorch does not become CUDA-capable merely because a GPU exists. Supplied Ultralytics checkpoints use CUDA when `torch.cuda.is_available()` succeeds. Training also selects CUDA, then Apple MPS, then CPU. Synchronize GPU timing for serious accelerator benchmarking; CPU timings cannot be extrapolated to GPU.

ONNXExecutor accepts an explicit execution-provider order. Its automatic order is TensorRT, CUDA, ROCm, MIGraphX, OpenVINO, DirectML, CoreML, XNNPACK, then CPU, restricted to providers reported by the installed runtime. CUDA, TensorRT and the other targets require their corresponding installed builds, libraries and supported graph operations. `/api/hardware/status` separates runtime availability from project validation. Record `session.get_providers()` and inspect profiling traces for per-node fallback. Provider availability is not proof that every operator ran there. Automatic cross-provider benchmarking and energy optimization still require target hardware.

## FPGA

`VitisExecutor` executes a compatible compiled XIR/VART single-DPU graph with caller-provided quantized tensors. Compilation, bitstream, quantization scales, input packing and supported operators are board-specific. The current adapter rejects mixed graphs. No FPGA is available in this workspace, so compatibility and speed are unverified. FPGA support is an inference integration interface, not arbitrary model training on an FPGA. Use CPU/GPU training and a validated vendor compilation pipeline.

## Jetson / Raspberry Pi

On Jetson use JetPack-compatible PyTorch/TensorRT packages and a board-tested container; never install generic desktop wheels indiscriminately. On Raspberry Pi use the CPU path or a compatible ONNX runtime, with small capture budgets. Four-stream throughput on either device is unverified. A Pi may be more useful as a camera gateway when perception cannot meet coverage requirements. Record actual board model, RAM, runtime versions, thermals, throughput and dropped frames.

## Containers

```powershell
$env:POSTGRES_PASSWORD = '<unique database password>'
$env:SURVEILX_ADMIN_PASSWORD = '<unique admin password>'
docker compose -f deployment/compose.yaml up --build
```

The Compose profile uses one API/worker process and PostgreSQL. It does not include research dependencies by default; build a separately pinned research image when training is needed. Optional S3-compatible evidence storage uses the standard AWS credential chain, `SURVEILX_S3_BUCKET`, and optional `SURVEILX_S3_ENDPOINT`. Evidence is application-encrypted in both storage modes. Back up the encryption key separately; losing it makes evidence unreadable.

Docker is unavailable on the current host, so container and PostgreSQL execution have not been verified. Local SQLite execution is a separate tested profile, not evidence of production database validation.

## Reliability and operations

- Independent source capture threads reconnect and retain bounded latest frames. Frame replacements are counted.
- One global inference loop allocates work; slow models cannot create unbounded camera queues.
- Detector failure falls back to the CPU baseline. Missing SVA-Net remains unavailable, not simulated.
- Failed incident database writes create encrypted spool files; evidence keys deduplicate replay.
- In-app notifications persist with incidents; provider integrations need configured delivery workers before use.
- Evidence expires according to `SURVEILX_RETENTION_DAYS`. Positive object sequences are recorded no more often than `SURVEILX_DETECTION_LOG_SECONDS` for an unchanged label set unless a configured incident rule records them first. The prototype retains audit metadata; telemetry partitioning and retention need deployment sizing.
- Run exactly one worker. Scaling requires external camera leases, cross-process notifications and shared rate limiting.
- Use TLS and secure cookies for remote access, restrict camera networks and storage IAM, protect data/key filesystem ACLs, and test restore procedures. Local source paths are confined to data/media; RTSP credentials are encrypted at rest.

## Python and C++ decision

OpenCV decoding, resizing, image operations and HOG execute in native code. PyTorch tensor kernels and ONNX Runtime execute in native CPU/accelerator libraries. Python owns application logic. Profile the full pipeline before creating a custom extension; a second native tracker or scheduler is unjustified without measured bottlenecks. The repository records microbenchmarks separately from end-to-end camera measurements. No custom C++ code is claimed or needed merely to satisfy a language preference.

The scratch detector's NMS was a measured Python bottleneck. It now uses TorchVision's native suppression kernel when available, with a Python fallback for unsupported builds/devices. `python -m scripts.benchmark_nms` records a seeded 600-box comparison and verifies retained-index parity; `reports/nms-benchmark.json` records the actual host result. This speedup applies to NMS, not the full detector. The tracker microbenchmark now uses 32 boxes rather than an empty detection list.
