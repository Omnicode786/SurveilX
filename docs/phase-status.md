# Phase status

Status as of 2026-09-30. This table distinguishes implemented code, executed evidence and remaining acceptance work. The full specification is a substantial development and deployment program; untested hardware, missing datasets or absent operator studies cannot be honestly marked complete.

| Phase | Implemented and exercised | Remaining work |
|---|---|---|
| 0 — Literature and novelty | Primary-source map, component provenance, concrete objectives and equations | Systematic exhaustive review and independently validated novelty claims |
| 1 — Infrastructure | Four generated streams, camera CRUD, local media/RTSP capture code, SQLite, encrypted filesystem evidence, authentication/RBAC, UI, WebSockets, incidents, in-app alerts, audit | Actual RTSP fault tests; PostgreSQL/S3 execution; external delivery workers; distributed leases/rate limiting |
| 2 — Baseline | HOG detector, synthetic fixture, IoU tracker, persistent-zone and configured site rules; pretrained YOLO11n fine-tuning and held-out AP50 | Tracking ID metrics, real event baselines and multi-domain evaluation |
| 3 — ASIE / RAIC | Camera environment configuration, motion/brightness state, CPU/RAM/telemetry governor, workload hysteresis | Learned semantic environment inference, evaluated contextual risk costs, calibrated energy optimization |
| 4 — YOLO-RAI | Backbone adapters, real pedestrian fine-tuning, independent calibration and test evaluation; runtime adapter | Supervised scene/zone risk conditioning, calibrated resolution variants, robust uncertainty under shift |
| 5 — SVA-Net | Independent scratch multiscale detector; entity-centric event model; configurable taxonomy and event contract; real detection and synthetic event training; scene-clip import/model with timestamp contract | Large-scale pretraining, real event labels, trained state/relation heads, multiple seeds/domains; scratch superiority not achieved |
| 6 — Arbitration | Semantic compatibility/calibration guards, disagreement review, secondary inference with exception isolation and abstention; durable named specialist slots; calibrated scene-event persistence policies and named incident evidence | End-to-end calibration of pooled evidence and evaluated multi-expert escalation on real cameras |
| 7 — MCRS | Global deadline-first scheduler, bounded service credit, overdue reservation, infeasibility reporting, fairness simulation | Hard real-time guarantees are not provided; measure competing real streams/model costs and event recall |
| 8 — Continuous adaptation | Evidence annotation UI and independent review; versioned train-only assembly; hard-example priority and bounded replay; opt-in automatic candidate training/calibration; frozen-model acceptance service; artifact-bound approval and durable canary/rollback; lineage-checked event/scratch warm starts and retention reports | Statistical drift-triggered retraining, real-site acceptance datasets, distributed workers, clustered confidence intervals and longitudinal old-domain retention evidence |
| 9 — Edge | Automatic CPU/CUDA/MPS selection, ONNX execution-provider interface, Vitis compiled-graph adapter, CPU export parity | Actual CUDA/TensorRT/FPGA/Jetson/Pi compilation, operation placement, thermal/power and throughput benchmarks |
| 10 — Evaluation | API/runtime/model tests, real pedestrian baseline comparison, generated event ablations, scheduler simulation, live generated-stream checks; D-Fire, SH17 and UR Fall development runs | Multi-domain transfer, rare-event recall, multi-seed confidence intervals, operator study, all hardware tiers; current hazard/event candidates are below deployment quality |
| 11 — Paper | Excluded by request | No paper manuscript is produced |

Detection and video-event tasks are the current scope. Segmentation, pose and audio heads are deferred. Format adapters never imply that unsupported supervision has been trained.

## Concrete external prerequisites

Finishing hardware validation requires access to the target accelerators/boards and compatible vendor runtimes. Real surveillance acceptance requires representative footage, reviewed task labels, source grouping and deployment criteria. Human-loop evaluation requires actual operators and a study protocol. The software records candidates and measurable outputs while these prerequisites remain unavailable.
