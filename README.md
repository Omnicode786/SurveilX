# SurveilX-Edge

A local command center for adaptive multi-camera monitoring, incident review and reproducible detection/video-event training. Works on an ordinary CPU computer, with optional accelerator integrations. This is a tested development prototype; deployment and research gaps are tracked explicitly in [phase status](docs/phase-status.md). The research paper is excluded.

## Run on Windows

```powershell
.\scripts\start.ps1
```

The script creates the Python environment when absent, builds the frontend when absent, and serves [the command center](http://127.0.0.1:8000). Python 3.11+ and Node/npm are required. The current environment was verified with Python 3.12 and Node 24.

Sign in as `admin`. On first startup, the generated password is saved locally in `data/initial-admin-password.txt`, unless configured explicitly. Keep `.env`, the encryption key, datasets and evidence private. The example environment enables four visibly marked generated streams for verification; connect RTSP, webcam or allowed local media sources through Cameras when ready.

If dependencies or frontend files change:

```powershell
.\.venv\Scripts\python -m pip install -e '.[dev,research]'
Push-Location frontend
npm ci
npm run build
Pop-Location
```

## Available workflows

- Live camera feeds show detected labels, counts, tracks and model provenance. Incident details explain what was detected and separate object detection from a reviewed incident decision.
- A hardware governor selects economy, balanced or performance workload budgets from measured resources. A global scheduler prioritizes overdue cameras and reports infeasible coverage.
- Incidents retain encrypted evidence, acknowledgment, review feedback and audit records. Four synthetic feeds exercise this workflow without implying real-world detection accuracy.
- Import labeled detection images or annotated event videos, validate independent splits, train, calibrate, evaluate and register a candidate. Choose the scratch architecture or modified YOLO in Experiments.
- Annotate incident evidence, obtain independent review, assemble later dataset generations with replay, and enable automatic candidate training. **Model acceptance** evaluates frozen models on independent data and records deployment criteria. See [reviewed learning](docs/reviewed-learning.md).
- Model activation records an explicit canary and previous version, survives restart and supports rollback. Production promotion requires real-domain acceptance.

## Models and data

**SVA-Detector** is a custom multiscale convolutional detector trained from random initialization. **SVA-Net** is an entity-centric video-event network. **YOLO11n with scene/zone adapters** is an explicitly inherited pretrained baseline. The present pedestrian benchmark does not establish that the scratch model outperforms YOLO.

Common detection imports support COCO boxes, YOLO detection labels and Pascal VOC boxes. Video-event data uses a declared frame/entity/image-size contract. New classes require compatible labeled training data; universal dataset compatibility and perfect accuracy are not promised.

Read [dataset compatibility](docs/dataset-adapters.md), [training workflow](docs/data-and-training.md), [mathematical foundations](docs/mathematical-foundations.md) and [real-image benchmark](docs/detection-benchmark.md). Public Penn-Fudan source files remain under ignored `data/`; provenance and redistribution restrictions are recorded in [dataset notes](docs/datasets.md).

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check surveilx training scripts tests
.\.venv\Scripts\python -m scripts.verify_live
```

The live verifier requires the server running and exercises generated streams, evidence, acknowledgment and feedback. It creates test audit activity. Reports are stored in `reports/`; training artifacts live in versioned `data/runs/` directories.

## Documentation map

| Topic | Reference |
|---|---|
| Implemented phases and remaining work | [Phase status](docs/phase-status.md) |
| System components | [Architecture](docs/architecture.md), [diagrams](docs/diagrams.md) |
| Objectives, losses, calibration, scheduler | [Mathematical foundations](docs/mathematical-foundations.md) |
| Data and later generations | [Adapters](docs/dataset-adapters.md), [training](docs/data-and-training.md) |
| Measurements and limitations | [Detection benchmark](docs/detection-benchmark.md), [evaluation](docs/evaluation.md) |
| CPU, CUDA, FPGA, containers | [Deployment](docs/deployment.md) |
| Prior art | [Literature and novelty map](docs/literature-and-novelty.md) |

Python orchestrates the application; OpenCV, PyTorch and ONNX Runtime already execute heavy operations in native libraries. Custom C++ should follow measured bottlenecks. CPU operation is verified here; CUDA, FPGA, Jetson and Raspberry Pi paths require target-specific builds and hardware validation. External notifications and distributed production deployment are unfinished.
