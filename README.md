# SurveilX-Edge

A local command center for adaptive multi-camera monitoring, incident review and reproducible detection/video-event training. Works on an ordinary CPU computer, with optional accelerator integrations. This is a tested development prototype; deployment and research gaps are tracked explicitly in [phase status](docs/phase-status.md). The research paper is excluded.

Current model/training status: **G8c completed nine jobs; G9 modified-YOLO-only training is running.** Live webcam inference is paused to free GPU resources. Admin Settings now supports performance preferences and explicit scoped model selection; production approval remains separate. The earlier general-object webcam baseline had no fire class. See [training status](docs/training-status.md) and [current continuation state](CONTINUE.md).

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

## Choose what to detect

Open **Camera wall**, select a camera, and use **What should this camera detect?** Turn categories or individual items on/off, then **Save detection choices**. Search finds specific items; Undo restores the last saved selection. Changes affect new detections and alerts, including fallback output. Existing recordings remain unchanged. A compatible active model is still required; turning an option on does not install or approve a model. Administrators and operators can edit; viewers can inspect.

Choices also control computation: disabled categories leave routing, unused specialist/event/fallback models skip inference, and cameras with no requested model work release their inference budget while previews continue. YOLO receives class filters before NMS; tracking and recording handle selected outputs only. Disabling a few classes in a shared neural network does not remove its shared backbone or guarantee higher FPS. See [selection performance](docs/detection-selection-performance.md) for measured limits.

The sidebar shows everyday pages first. **More tools**, **Detection details** and **Advanced alert rules** keep technical controls available without crowding the main view.

## Live performance tuning

Camera previews use authenticated continuous MJPEG with shared, bounded JPEG encoding and disconnect when the tab is hidden. Performance mode requests30captureFPS (balanced15/economy5); actual delivery depends on the camera, lighting and system load. Detection runs independently; preview FPS is not model FPS. Preview JPEGquality80 does not alter stored raw evidence or model inputs. Evidence/input-motion sampling is capped at8Hz to retain bounded memory and a useful review time window. Expensive statistical drift checks run in one background worker.

For a supported CUDA environment, a responsive local configuration is `SURVEILX_EPOCH_SECONDS=0.25` and `SURVEILX_RUNTIME_BUDGET_SCALE=1.75` before starting the server. Set `SURVEILX_POWER_PROFILE=performance` to request the highest profile on a GPU computer with sufficient RAM; thermal, memory, latency and low-battery throttling still apply. The scale defaults to 1 and accepts 0.25–2; the total scheduling budget is capped at 900 ms per second. Hardware pressure still reduces the workload. Keep one server worker for physical cameras. Use the installed CUDA environment and existing approved or explicitly configured local-test detector; these settings do not approve models or start training. Actual throughput depends on hardware, power state, inference cost and evidence storage.

## Available workflows

- Live camera feeds show detected labels, counts, tracks and model provenance. Incident details explain what was detected and separate object detection from a reviewed incident decision.
- A hardware governor selects economy, balanced or performance workload budgets from measured resources. A global scheduler prioritizes overdue cameras and reports infeasible coverage.
- Incidents retain encrypted raw frames, titled review frames, raw/review clips, acknowledgment, review feedback and audit records. Positive detections are logged at a bounded interval even outside a configured rule, and every review frame states what was detected. Four synthetic feeds exercise this workflow without implying real-world detection accuracy.
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

## Team learning - click your assigned files

This is the **code-reading plan for Ashar, Abdur Rehman, Ayaan and Owais**. Each file name below is a clickable link that opens the actual source or document in this repository. **Read in the listed order**; be ready to explain what the file receives, what it does, what it returns, and show one working example to the instructor.

**Start here (everyone):** [Project README](README.md) -> [System diagrams](docs/diagrams.md) -> [Architecture](docs/architecture.md) -> [Implemented vs. pending features](docs/phase-status.md) -> [Current training evidence and limitations](docs/training-status.md).

For explanations, the seven-session timetable, demo practice, and instructor viva questions, open the **[complete four-person learning and instructor defense guide](TEAM_LEARNING_GUIDE.md)**.

### Abdur Rehman - Architecture, backend and API integration

**Read in order:**
1. [System diagrams](docs/diagrams.md) - follow the full camera-to-dashboard flow.
2. [Architecture specification](docs/architecture.md) - learn why each module exists.
3. [Configuration: `surveilx/config.py`](surveilx/config.py) - learn application settings and local defaults.
4. [FastAPI endpoints: `surveilx/api.py`](surveilx/api.py) - follow login, cameras, incidents, and WebSocket routes.
5. [Runtime integration: `surveilx/runtime.py`](surveilx/runtime.py) - see how API actions relate to running cameras.
6. [Experiment jobs: `surveilx/jobs.py`](surveilx/jobs.py) - understand how offline training jobs are managed.
7. [API tests: `tests/test_api.py`](tests/test_api.py) - show a real tested request.

**Instructor task:** Draw the main architecture and trace *one* UI request through FastAPI to its response.

### Ashar - Database, authentication, incidents and evidence

**Read in order:**
1. [Database models: `surveilx/database.py`](surveilx/database.py) - users, sessions, cameras, incidents, alerts, audit.
2. [Security: `surveilx/security.py`](surveilx/security.py) - password hashing, sessions, roles, encryption keys.
3. [Incident lifecycle: `surveilx/incidents.py`](surveilx/incidents.py) - legal state changes and invalid-transition checks.
4. [Evidence handling: `surveilx/evidence.py`](surveilx/evidence.py) - saved frames/clips and encrypted bundles.
5. [Notifications: `surveilx/notifications.py`](surveilx/notifications.py) - in-app alerts, grouping, acknowledgment and escalation.
6. [Reviewed learning guide](docs/reviewed-learning.md) - understand why model predictions are not human-reviewed ground truth.
7. [Security and incident tests: `tests/test_api.py`](tests/test_api.py) - follow role checks and incident-state tests.

**Instructor task:** Explain how a detection becomes a *reviewable* incident, how evidence is protected, and why an alert is not proof of a real event.

### Ayaan - React dashboard, frontend and live data

**Read in order:**
1. [Frontend dependencies: `frontend/package.json`](frontend/package.json) - React, TypeScript, Vite and build commands.
2. [Main dashboard: `frontend/src/main.tsx`](frontend/src/main.tsx) - navigation, API calls, login and WebSocket updates.
3. [Detection display: `frontend/src/detections.tsx`](frontend/src/detections.tsx) - labels, counts, confidence and evidence.
4. [Incident rules UI: `frontend/src/policies.tsx`](frontend/src/policies.tsx) - configuring camera policies.
5. [Learning panels: `frontend/src/learning.tsx`](frontend/src/learning.tsx) - annotation and reviewed learning screens.
6. [Performance view: `frontend/src/performance.tsx`](frontend/src/performance.tsx) - model and training metrics.
7. [Alerts UI: `frontend/src/notifications.tsx`](frontend/src/notifications.tsx) - notification settings and behavior.

**Instructor task:** Demo login -> overview -> a **clearly labeled generated demo feed** -> detections/incident details, and explain how React receives HTTP/WebSocket data.

### Owais - Cameras, scheduling, AI models and evaluation

**Read in order:**
1. [Camera processing: `surveilx/runtime.py`](surveilx/runtime.py) - capture, live frames, inference and camera health.
2. [Detector and tracker: `surveilx/vision.py`](surveilx/vision.py) - HOG baseline, YOLO integration and tracking.
3. [Global scheduler: `surveilx/controller.py`](surveilx/controller.py) - deadlines, priorities, inference cost and budgets.
4. [Hardware governor: `surveilx/hardware.py`](surveilx/hardware.py) - CPU/RAM resource-aware operation.
5. [Incident rules engine: `surveilx/policies.py`](surveilx/policies.py) - observations, zones and review candidates.
6. [Scratch detector: `training/detector_model.py`](training/detector_model.py) - independent multiscale detector.
7. [SVA v3: `training/detector_v3.py`](training/detector_v3.py) - revised four-scale detector architecture.
8. [Video-event models: `training/models.py`](training/models.py) - SVA-Net and clip reasoning.
9. [Adapted pretrained YOLO: `training/yolo_model.py`](training/yolo_model.py) - scene/zone adapters; **not** scratch YOLO.
10. [Evaluation methods](docs/evaluation.md) and [runtime tests](tests/test_runtime.py) - know what was measured and how.

**Instructor task:** Walk through one frame from capture to detector to scheduler/rules, and distinguish SVA-Detector, SVA-Net and pretrained YOLO.

**Team rule:** Studying these files does **not** require changing code, starting new model training, activating candidate models, or presenting simulated results as real-camera accuracy.

## Documentation map

| Topic | Reference |
|---|---|
| Implemented phases and remaining work | [Phase status](docs/phase-status.md) |
| System components | [Architecture](docs/architecture.md), [diagrams](docs/diagrams.md) |
| Objectives, losses, calibration, scheduler | [Mathematical foundations](docs/mathematical-foundations.md) |
| Data and later generations | [Adapters](docs/dataset-adapters.md), [training](docs/data-and-training.md) |
| Measurements and limitations | [Detection benchmark](docs/detection-benchmark.md), [evaluation](docs/evaluation.md) |
| Category routing and current comparisons | [Model selection](docs/model-selection.md), [results](docs/selection-results-1015.md) |
| CPU, CUDA, FPGA, containers | [Deployment](docs/deployment.md) |
| Prior art | [Literature and novelty map](docs/literature-and-novelty.md) |

Python orchestrates the application; OpenCV, PyTorch and ONNX Runtime already execute heavy operations in native libraries. Custom C++ should follow measured bottlenecks. CPU operation is verified here; CUDA, FPGA, Jetson and Raspberry Pi paths require target-specific builds and hardware validation. External notifications and distributed production deployment are unfinished.
