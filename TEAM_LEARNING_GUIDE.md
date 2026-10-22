# SurveilX-Edge | Team Learning & Instructor Defense Guide

**Audience:** Ashar, Abdur Rehman, Ayaan and Owais  
**Purpose:** Understand the *existing* SurveilX repository, follow its data flow, locate the important code, demonstrate the app and confidently explain it to an instructor.  
**Repository:** [Omnicode786/SurveilX](https://github.com/Omnicode786/SurveilX)  
**Guide version:** 22 October 2026 | **Study plan:** 7 sessions | **Rule:** Understand first; do not rewrite or retrain the project just to complete this guide.

> **Important status note.** SurveilX is a working **local development prototype**, not a fully validated commercial surveillance solution. The repository's dated training documents describe a G8c training campaign and changing model results; they do **not** establish today's live job status or general real-world accuracy. Check `docs/training-status.md` and the local app before quoting experiment status. Some acceleration, deployment, external notifications and real-site evaluations remain unfinished. Do not call a generated stream a real camera or say that synthetic scores prove field accuracy.

## 1. Understand the complete project in 90 seconds

SurveilX-Edge is a **local multi-camera monitoring command center**. It can take frames from permitted camera/video sources or clearly labeled demo feeds, run available visual detectors, track observed objects, apply configured scene/zone rules, create reviewable incidents, save evidence, notify an authorized operator and show everything in a React dashboard. A separate **research workflow** handles labeled datasets, model training, evaluation, candidate registration and guarded approval. A **hardware governor and global scheduler** decide how much inference work the computer can afford across cameras.

**Technology:** Python + FastAPI (API/orchestration); OpenCV (capture and image processing); PyTorch / Ultralytics / ONNX Runtime (model experiments and inference integrations); SQLAlchemy + SQLite for the local profile; React + TypeScript + Vite (dashboard); WebSockets (live updates). See [`README.md`](README.md), [`pyproject.toml`](pyproject.toml) and [`frontend/package.json`](frontend/package.json).

**Explain the flow using these nine steps:**

1. **Capture** - `surveilx/runtime.py` reads a real source or creates an explicitly labeled demo frame.
2. **Observe** - frame properties, tracking history, configured camera environment and hardware telemetry are collected.
3. **Allocate** - `surveilx/controller.py` selects camera work within a global inference budget.
4. **Detect** - `surveilx/vision.py` and model adapters return objects, labels, boxes and confidence values.
5. **Interpret** - `surveilx/policies.py` evaluates configured evidence and zones; available video-event experts can add separate evidence.
6. **Create incident** - `surveilx/incidents.py` manages reviewable incident state transitions.
7. **Store evidence** - `surveilx/evidence.py` writes encrypted evidence bundles; `surveilx/database.py` holds application records.
8. **Notify and display** - `surveilx/notifications.py` handles in-app alerts, while `surveilx/api.py` exposes HTTP/WebSocket data to React.
9. **Review and improve** - an authorized person can review evidence/labels; `surveilx/adaptation.py` and `surveilx/acceptance.py` govern later candidate evaluation. A model is **not** automatically production-approved because it trained.

**Mental model:** `Cameras -> Capture -> Scheduler -> Detector / Event Expert -> Rules -> Incident + Evidence -> API/WebSocket -> Dashboard -> Human Review`. Offline training is a **separate branch**, not a magic self-learning loop.

## 2. What everyone must read first

Read in this exact order, regardless of your assigned module:

| Order | Source | What you should understand |
|---|---|---|
| 1 | [`README.md`](README.md) | Project objective, how to run, main workflows and honesty about limitations |
| 2 | [`docs/diagrams.md`](docs/diagrams.md) | Overall system, incident lifecycle, scheduler and data movement |
| 3 | [`docs/architecture.md`](docs/architecture.md) | Why each component exists and which part owns each decision |
| 4 | [`docs/phase-status.md`](docs/phase-status.md) | What is implemented vs. unfinished or only simulated |
| 5 | [`docs/training-status.md`](docs/training-status.md) | What metrics and experiment statuses may actually be reported |

**Group rule:** You do not need to memorize every Python class. You **do** need to answer (a) what comes in, (b) what the module does, (c) what comes out, (d) what file implements it, and (e) how to prove it using a test or demo.

## 3. Assigned ownership: exactly four members

| Member | Primary module to study | Key files | Instructor responsibility |
|---|---|---|---|
| **Abdur Rehman** | Architecture, backend API and integration | `surveilx/api.py`, `surveilx/config.py`, `surveilx/jobs.py`, `docs/architecture.md` | Explain complete system and API-to-runtime flow |
| **Ashar** | Database, authentication, incidents and evidence security | `surveilx/database.py`, `surveilx/security.py`, `surveilx/incidents.py`, `surveilx/evidence.py`, `surveilx/notifications.py` | Explain access control and proof of incidents/evidence |
| **Ayaan** | Dashboard, user interaction and live updates | `frontend/src/main.tsx`, `frontend/src/detections.tsx`, `frontend/src/learning.tsx`, `frontend/src/policies.tsx`, `frontend/src/performance.tsx` | Drive the live UI demo and explain React-to-API communication |
| **Owais** | Camera pipeline, scheduling and AI model overview | `surveilx/runtime.py`, `surveilx/controller.py`, `surveilx/hardware.py`, `surveilx/vision.py`, `training/detector_model.py`, `training/models.py` | Explain frames, inference budget, scratch/YOLO/SVA-Net distinctions |

### Abdur Rehman - Backend, API & architecture (lead for the big picture)

**Study in order:** `docs/diagrams.md` -> `docs/architecture.md` -> `surveilx/config.py` -> `surveilx/api.py` -> `surveilx/runtime.py` (integration only) -> `surveilx/jobs.py`.

**Learn these topics:** Why one local FastAPI application serves the command center; why SQLAlchemy and the capture runtime are separate; what an HTTP REST endpoint does; when WebSockets provide updates; how a camera is registered; where the startup/shutdown lifecycle manages the runtime; how API requests reach database/security/runtime functions; and why training jobs run separately from the live detection loop.

**Trace one real code path:** In `surveilx/api.py`, follow `POST /api/auth/login`, `GET /api/cameras`, `POST /api/cameras`, `GET /api/incidents` and a live WebSocket route. Then locate where a created camera is synchronized with the running capture system. Do not say the API itself runs a neural network in the browser.

**Mini assignment:** Draw a 6-box backend architecture diagram. For **one** route, note HTTP method, input, authentication, action and response. Find a related test in `tests/test_api.py`.

**Two-sentence instructor explanation:** “FastAPI is the entry point and controls the server-side workflows. It validates authenticated requests and connects the dashboard to the camera runtime, database, incidents and experiments, while keeping heavy inference out of the HTTP request handler.”

**Be ready for:** What is FastAPI? Why is it modular? REST vs WebSocket? What runs at startup? What happens if a camera disconnects? Why is production scaling different from a one-process development server?

### Ashar - Security, database, evidence & incident lifecycle

**Study in order:** `surveilx/database.py` -> `surveilx/security.py` -> `surveilx/incidents.py` -> `surveilx/evidence.py` -> `surveilx/notifications.py` -> `docs/reviewed-learning.md`.

**Learn these topics:** Core SQLAlchemy tables (`User`, `Session`, `Camera`, `Incident`, `Alert`, `Audit`, `Feedback`, `ModelVersion`); password hashing and expiring sessions; role-based authorization; encryption of camera-source credentials and saved evidence; audit trails; incident states such as VERIFYING, ACKNOWLEDGED and RESOLVED; how in-app alerts are grouped and acknowledged; and why reviewed human labels are distinct from model predictions.

**Trace one real code path:** Authentication in `surveilx/security.py` -> endpoint permission in `surveilx/api.py` -> `create_incident` / `transition` in `surveilx/incidents.py` -> saved evidence in `surveilx/evidence.py` -> notification record. Use `tests/test_api.py` to observe an invalid incident transition returning HTTP 409.

**Mini assignment:** Sketch a tiny database relationship diagram: `User -> Session`; `Camera -> Incident`; `Incident -> Alert / Feedback`; `Audit` records actions. Show the difference between an **alert** and a **confirmed real-world event**.

**Two-sentence instructor explanation:** “SurveilX stores camera and incident metadata in a database and keeps protected evidence separately. Users have roles, critical actions are audited, and incidents follow controlled transitions so an AI observation is not automatically treated as a verified accusation.”

**Be ready for:** Why hash passwords? What is an HttpOnly cookie? What is RBAC? What is encrypted and why? Why not store video blobs directly in every database row? What is a false positive? What happens when an operator acknowledges an alert?

### Ayaan - React command center & demonstration

**Study in order:** `frontend/package.json` -> `frontend/src/main.tsx` -> `frontend/src/detections.tsx` -> `frontend/src/policies.tsx` -> `frontend/src/learning.tsx` -> `frontend/src/performance.tsx` -> `frontend/src/notifications.tsx`.

**Learn these topics:** React components, state and effects; navigation between overview/cameras/incidents/experiments; the frontend `api()` helper; `fetch('/api/...')`; session-based requests; WebSocket reconnection; display of camera streams; detection label/score/track ID; differences between an incident and a model estimate; and how learning/performance panels show results without claiming every task is solved.

**Trace one real code path:** A click in the UI -> frontend request -> matching `surveilx/api.py` route -> updated response -> React component re-render. For live data, locate the WebSocket connection and its update handler in `frontend/src/main.tsx`.

**Mini assignment:** Prepare a 3-minute safe demo: login -> overview -> one **generated demo** camera -> detection details -> incidents/review -> hardware/models/performance. If a screen needs real labels or a trained model, say so rather than presenting mock values as actual detection.

**Two-sentence instructor explanation:** “React provides the command center that operators use to monitor cameras, inspect detected labels and review incidents. The UI is not the AI engine: it calls the FastAPI backend and receives live updates over WebSockets.”

**Be ready for:** Why React? What is component state? HTTP request vs socket update? Which component shows detections? Where are incidents fetched? How would a user distinguish demo streams and real evidence?

### Owais - Camera runtime, AI inference, scheduling & limitations

**Study in order:** `surveilx/runtime.py` -> `surveilx/vision.py` -> `surveilx/controller.py` -> `surveilx/hardware.py` -> `surveilx/policies.py` -> `training/detector_model.py` -> `training/detector_v3.py` -> `training/models.py` -> `training/yolo_model.py` -> `docs/evaluation.md`.

**Learn these topics:** OpenCV frames and latest-frame capture; reconnect/backpressure and object detection vs object tracking; HOG as a baseline and generated fixtures as test data; global inference budget; what the `Scheduler.allocate()` prioritizes; why a CPU/RAM governor adjusts workload; how policies turn *observations* into review candidates; and the distinction among **SVA-Detector**, **SVA-Net** and **modified pretrained YOLO**.

**Models - simple explanation:** `training/detector_model.py` implements the independently initialized scratch object detector; `training/detector_v3.py` contains a later four-scale version. `training/models.py` implements video-event research networks (including entity-centric `SVANet`), not an object detector. `training/yolo_model.py` adds experimental scene/zone adapters to an **inherited pretrained** YOLO backbone, so it is not trained fully from scratch. Actual model performance must be read from held-out evaluation reports and experiment manifests, **not** inferred from the architecture name.

**Trace one real code path:** `Capture` receives a frame -> `Runtime` schedules inference -> `Scheduler.allocate()` selects work -> vision detector produces boxes -> tracker/rules review observations -> evidence/incident may be recorded. Check `tests/test_runtime.py` for a scheduler case and synthetic stream verification.

**Mini assignment:** Explain `deadline`, `priority`, `cost_ms`, `budget_ms` with a 2-camera example. Show the difference between AP50, event accuracy, false positives and real-site validation. **Do not launch training, activate a model or import new datasets as a study exercise.**

**Two-sentence instructor explanation:** “The runtime reads camera frames and applies the available detector under a controlled hardware budget. The scheduler shares limited inference time across cameras, while experimental scratch, YOLO and video-event models are separately evaluated before any activation.”

**Be ready for:** Detection vs tracking? Scratch vs pretrained? Why can a CPU-only PC run the prototype? What is a frame? What happens when inference is slow? Why does AP50 not mean 80% incident accuracy?

## 4. Seven-session learning timetable

| Session | All four members do | Individual evidence to produce |
|---|---|---|
| **1 - Overview** | Read README and system diagrams; explain the complete camera-to-dashboard flow | Each draws the same flow from memory |
| **2 - Core files** | Read architecture and assigned code; identify key classes/functions | Each makes a 1-page file/function map |
| **3 - Follow a request** | Trace one input all the way to a result | One route-to-output trace per person |
| **4 - Hands-on** | Run the local app safely and observe **generated** feeds | Screenshot or notes showing one real UI behavior |
| **5 - Evidence and tests** | Read own related tests and phase-status limitations | One passed/reproducible test or code walkthrough |
| **6 - Teach back** | Each gives a 4-minute explanation; teammates ask questions | Correct technical vocabulary + 3 instructor answers |
| **7 - Mock defense** | Do an uninterrupted 12-15 minute joint demonstration | Final checklist signed by all four |

**Recommended reading pace:** 60-90 minutes per session per member, with a short 15-minute team sync. Rotate as needed; the seven sessions are an onboarding guide, not claims about work already completed.

## 5. Instructor presentation: 12-minute script

| Time | Speaker | What to show or say |
|---|---|---|
| 0:00-2:00 | **Abdur Rehman** | Problem statement, technologies, overall architecture and camera-to-dashboard flow |
| 2:00-4:30 | **Owais** | Capture, detector, scheduling, scratch vs YOLO vs event model, honest limitations |
| 4:30-7:00 | **Ashar** | Database, session security, incident lifecycle, evidence and human review |
| 7:00-10:30 | **Ayaan** | Live UI tour with generated feeds clearly identified, incidents and performance panels |
| 10:30-12:00 | **All** | Summarize contributions, tests, known limits and future improvements; take questions |

**One-line handoffs:** Abdur Rehman -> “Owais will explain how frames are processed.” Owais -> “Ashar will explain what happens to detections and their evidence.” Ashar -> “Ayaan will show how an operator sees all this.” Ayaan -> “Now we will answer questions about the code and results.”

## 6. Safe local demo and code checks

On **Windows PowerShell**, from the repository root:

```powershell
.\scripts\start.ps1
```

Visit `http://127.0.0.1:8000`. Follow README's first-run admin-credential instructions **on the local machine**; never paste the password, `.env`, keys, camera URLs or private footage into slides/screenshots.

For an existing configured virtual environment, a developer can inspect these checks without starting new model training:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check surveilx training scripts tests
Push-Location frontend
npm run build
Pop-Location
```

Tests can create or modify local **test** records, so use a test environment, not production evidence. A saved test report is historical evidence; a new run should report its own result. If dependencies are missing, use README setup instructions instead of guessing a success result.

### Five proof points for instructor questions

1. **Real code path:** Open `surveilx/api.py` and explain one authenticated endpoint.
2. **Real live data path:** Open the WebSocket handling in `frontend/src/main.tsx` and show how it updates the UI.
3. **Real scheduling logic:** Open `Scheduler.allocate()` in `surveilx/controller.py` and explain deadline, cost and budget.
4. **Real incident controls:** Open `TRANSITIONS` in `surveilx/incidents.py` and show why invalid transitions are rejected.
5. **Real validation boundary:** Open `docs/phase-status.md` and `docs/training-status.md` and describe what is still unverified.

## 7. Instructor questions - short, accurate answers

**Q1. What problem does SurveilX solve?** It combines multiple camera feeds, resource-aware detection and incident review in one local command center.

**Q2. Where is the application's main backend?** `surveilx/api.py` exposes FastAPI routes; `surveilx/runtime.py` handles long-running camera/inference work.

**Q3. Does the frontend perform object detection?** No. React renders data; the backend/runtime and model integrations process frames.

**Q4. How do live updates reach the UI?** Through a WebSocket connection, in addition to ordinary HTTP requests.

**Q5. Is SVA-Net the same as SVA-Detector?** No. SVA-Detector predicts object boxes/classes; SVA-Net studies temporal video-event evidence from clips/entities.

**Q6. Did you create YOLO from scratch?** No. The repository uses an inherited pretrained YOLO and experiments with scene/zone adapters. The separate SVA detector is the scratch architecture.

**Q7. Why use a scheduler?** Several cameras compete for limited CPU/GPU time. The scheduler selects affordable inference tasks while tracking coverage limits.

**Q8. How do incidents get verified?** Model detections and configured rules create reviewable observations; authorized humans can acknowledge, label or resolve incidents. A detection alone is not a proven real-world event.

**Q9. Is the database encrypted?** Camera-source secrets and evidence have application-level encryption; do not claim the entire database has full-disk encryption by default.

**Q10. Is SurveilX fully deployed to production hardware?** No. The repository distinguishes local tests and integration interfaces from untested hardware or production services.

**Q11. What does AP50 mean?** Average precision for object detection at an intersection-over-union threshold of 0.50. It is not the percentage of security incidents correctly understood.

**Q12. What is the research contribution?** A tested *combination and evaluation* of multi-camera orchestration, adaptive scheduling, model options and human-review workflows. Architectural originality and field superiority require independent evidence; do not claim them without it.

## 8. Final readiness checklist

- [ ] All four can draw the nine-step system flow without notes.
- [ ] Each person can name at least four **verified** code files in their area and show one function.
- [ ] Each can explain a feature's input, action, output, limitation and proof.
- [ ] Ayaan can demonstrate the interface without exposing secrets or mislabeling demo feeds.
- [ ] Owais can distinguish the scratch detector, event model and adapted pretrained YOLO.
- [ ] Ashar can explain states, RBAC, audit and evidence encryption accurately.
- [ ] Abdur Rehman can trace REST + WebSocket through to the backend and runtime.
- [ ] Every experimental metric is tied to its specific task, dataset, checkpoint and evaluation split.
- [ ] Group can answer at least eight of the twelve instructor questions.
- [ ] Screenshots and slides show **existing behavior**, not promises.

**Finish line:** The team can collectively explain *what SurveilX currently does, how the code accomplishes it, how it was tested, and what remains unproven*.

---

**Further reading:** [`docs/deployment.md`](docs/deployment.md) | [`docs/evaluation.md`](docs/evaluation.md) | [`docs/reviewed-learning.md`](docs/reviewed-learning.md) | [`docs/mathematical-foundations.md`](docs/mathematical-foundations.md) | [`docs/literature-and-novelty.md`](docs/literature-and-novelty.md)