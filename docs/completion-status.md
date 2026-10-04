# Completion status

Verified 2026-10-04. The app is a working development system; the full requested multi-domain deployment is incomplete. A single completion percentage would mix implemented code, running experiments and unvalidated field performance. The research paper is excluded.

Current scope also excludes complete acceleration deployment, full hardware acceleration coverage, complete production verification and full production infrastructure/deployment completion. Those historical gaps below are deferred, not active tasks. The latest user request separately authorizes local CUDA training on the laptop GPU.

| Area | Complete now | Still needed |
|---|---|---|
| Core application | Camera/stream management, dashboard, authentication/roles, named detections, tracking, incident review, encrypted titled frame/clip logs; grouped role-routed in-app alerts with priority overrides, acknowledgement and timed escalation | Real-camera and production database/storage/worker deployment validation |
| Learning workflow | Generated/public/user data import, separate train/validation/calibration/test splits, reviewed annotations, scratch/adapted models, candidate training, calibration, lineage, acceptance/canary/rollback | More representative labels and independent site/participant acceptance datasets |
| Model generations | Twelve-model accuracy campaign, PPE scratch retry and nine detector-profile jobs complete; later weapons/fire/PPE/fighting candidates evaluated; six new CUDA candidates running/queued | All-class 50% target remains unmet; current g4 results are not yet evaluated |
| Event intelligence | Scene-video importer, temporal model/runtime, corrected persistence policies; real fall candidate; fighting model trained/calibrated/evaluated on 160 clips with paired-camera groups | Independent fighting evaluation and temporal label review; theft and collision data/training; broader fall evaluation |
| Policies | Restricted zones, wrong-way, possible missing PPE, proximity, calibrated speed, signal stop-line, machine state and blocked exit | Site geometry, trained signal/state labels, real footage and operator validation |
| Hardware | CPU and local CUDA training exercised; scratch/adapted YOLO/scene optimizer checks and measured batch tuning pass; live GPU memory/utilization displayed; DirectML exports preserve validation AP50; 11 provider/governor scenarios simulated | Physical FPGA/Jetson/Pi/AMD/Apple execution and accelerated deployment remain explicitly deferred |

## Latest held-out development results

AP50 is a detection metric, not the percentage of frames classified correctly.

| Family / model | Test result | Interpretation |
|---|---:|---|
| Person / adapted YOLO | AP50 0.9427 | Narrow pedestrian development benchmark |
| Person / scratch | AP50 0.7177 | Scratch does not outperform adapted YOLO |
| Dangerous items / adapted YOLO | AP50 0.5041 | Aggregate point estimate passes; knife/machete remain below 0.50; uncertainty interval crosses 0.50 |
| Fire/smoke / adapted YOLO | AP50 0.4785 | Fire 0.5156; smoke 0.4414 |
| PPE / adapted YOLO | AP50 0.3056 | Helmet 0.6299; safety vest 0.3197; multiple rare classes weak |
| Fighting scene model | 14/20 clips correct (0.70) | Both classes F1 0.70; group-bootstrap 95% interval 0.50-0.90; staged single-room weak labels |
| PPE / scratch | AP50 0.0527 | Completed retry remains weak |
| Fall scene model | 10/10 clips correct | Wilson 95% bound 0.7225–1.0 for ten all-correct groups, conditional on independence; shared-site uncertainty remains |

All 50 completed artifact weight hashes passed the integrity audit; both interrupted scratch originals have complete replacements. Newly trained development candidates remain inactive and deployment-ineligible. Inherited general-object class coverage is separate from hazard/event recognition.

## Accuracy work that can proceed now

1. Local CUDA training is configured and verified on the MX130. Real-sample training-step measurements show 3.21x scene, 5.65x scratch and 4.66x adapted-YOLO speedups, excluding loading/full-epoch overhead. A completed GPU fighting continuation retained its stronger parent and remained at 0.70 accuracy; no accuracy gain is claimed.
2. Preserve the completed PPE retry and all nine profile results. Both interrupted scratch jobs already have verified completed replacements; do not restart completed experiments blindly.
3. Use corrected diagnostics at each model's real input size and sample across classes. The weapons scratch probe finds 16/20 validation objects localized at IoU 0.50, but only 5 of those with the correct class, on a bounded stratified subset. Classification and score quality deserve controlled training ablations; merely increasing resolution is not a demonstrated remedy. This subset is diagnostic, not an accuracy estimate.
4. Improve rare-class/source coverage and evaluate on independent footage. Additional training alone cannot establish support for missing theft/collision labels or unreviewed fighting intervals or unseen sites.

Latest application improvements cover finite stop-line crossings, uninterrupted event persistence, exact video endpoints, immutable import settings, weak-label disclosure and group-aware event uncertainty. Native sequential video sampling matched repeated-seek pixels and took 0.87 versus 5.55 seconds on one 1080p clip; broader speed claims are not established. The detailed resumable state is in `CONTINUE.md`.

## Current follow-up

The local suite passed 174 tests, repository Ruff passed and the frontend build passed (`index-DelKl4Zt.js`). Notification policy and REST/live recipient filtering are exercised; external channels are unconfigured. See [notifications.md](notifications.md).

The charger-powered MX130 batch-tuning report selected YOLO 8, scratch 4 and scene 8 within its reserved-memory rule. Actual YOLO training reaches 99% GPU utilization with approximately 400-450 MiB free VRAM. Heavy GPU jobs run serially. The larger D-Fire training set contains 1,284 images; original 96-image validation, calibration and test records remain unchanged. The g4a/g4b queues cover PPE, weapons, fire/smoke and scratch focal-classification ablations. Their pending results are not included in the completed-score table above.
