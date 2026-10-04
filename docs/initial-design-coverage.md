# Initial design coverage

Verified local development snapshot: 2026-10-04. This is a product checklist, not a paper. Implementation, training, measured development results and independent domain acceptance are separate milestones.

| Original design area | Implemented / exercised | Still incomplete within current scope |
|---|---|---|
| Acquisition, review and evidence | Generated, local-media and RTSP inputs; reconnect fault check; encrypted raw and titled detection frames/clips; audit and incident state machine | Representative reviewed real-camera fault and event footage |
| Alerts and operator review | Persisted in-app policy, incident/notification grouping, event priorities, role routing, acknowledgement and timed escalation; REST/live filtering tests | Operator usability and real-site escalation validation |
| Detection baselines and tracking | Pretrained/adapted YOLO, scratch multiscale detection, class-aware IoU tracker, calibration and held-out AP50 | Tracking identity metrics on labeled sequences; domain-specific full-taxonomy evaluation |
| Environment and risk | Configured camera environments, motion/brightness context, workload governor, supervised adapter inputs and explicit site policies | Learned semantic environment inference, reviewed scene/zone/state/relation targets and evaluated risk costs |
| Scratch architecture | Scratch object detector and entity/scene temporal networks; reviewed manifest contracts; from-scratch training and lineage-checked continuation | Reliable hazard classification, larger representative training corpora, multiple seeds and evidence of improvement over YOLO |
| Multiple experts | Named specialist slots, calibrated compatibility guards, exception isolation, abstention and disagreement review | Real-camera pooled-evidence calibration and comparative event recall |
| Camera scheduling | Deadline-first scheduling, bounded credit, overdue reservation, infeasibility reporting and fairness simulation | Measured competing real streams and the effect on rare-event recall |
| Continuous learning | Review-gated annotations, train-only version assembly, replay, opt-in candidate training/calibration, grouped uncertainty, drift review, acceptance and canary/rollback gates | Independent site labels, drift-threshold validation and longitudinal retention |
| Model efficiency | Nine fitted resolution profiles; actual local CUDA training and batch tuning; actual DirectML export parity; explicit provider simulations | Accuracy-preserving improvements on weak scratch/PPE/smoke/weapon classes; larger user contracts require fresh measurements |
| Multi-domain hazards/events | Real-source development candidates for fire/smoke, dangerous items, PPE, falls and fighting; traffic/industrial site-policy primitives | Theft/collision footage and trained candidates; reviewed signal/machine-state labels; broader industrial evidence; every-class 0.50 target and independent field reliability |

The current CUDA queues cover PPE and weapons adapted YOLO, expanded fire/smoke adapted YOLO, and focal-classification trials for the three weak scratch families. Checkpoints are selected using validation; calibration and test reporting follow training. Stronger parents are retained when validation does not improve. These jobs do not fill missing theft/collision supervision or establish independent reliability.

The D-Fire expansion preserves original validation/calibration/test records and adds 900 train-only images. Its filename-band groups are development proxies, not verified independent scenes. AIRTLab fighting uses staged one-room footage with inherited video labels. Existing fall results are based on ten test groups. These limitations remain visible in capability and readiness reports.

Research paper, complete acceleration deployment, full hardware acceleration coverage, complete production verification and full production infrastructure/deployment remain excluded by the user. External alert providers and distributed production delivery workers belong to that deferred scope.

Evidence: `reports/product-readiness.json` verifies 50 completed weight hashes; `reports/gpu-batch-api-verification.json` records authenticated local checks and current processes; `data/hardware/cuda-batch-tuning-g1/report.json` records batch measurements. The full Python suite passed 174 tests, repository Ruff passed and the TypeScript/Vite build passed. Exact running and queued job identities are maintained in `CONTINUE.md`.
