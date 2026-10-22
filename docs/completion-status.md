# Completion status

> Current status (2026-10-15): user reauthorized training after the diagnosis. Capacity-tuned G8c is running nine serial CUDA jobs with new outputs and frozen sources; gains are not yet established. No learned model is activated. See [training status](training-status.md) and [diagnosis](g7-stagnation-diagnosis.md).

Verified 2026-10-05. SurveilX is a working local development application. Its multi-domain models are not yet reliable enough for full operational acceptance. Code completion and measured model quality are separate milestones; no single completion percentage represents both.

**Current scope:** controlled training/evaluation and software improvements are authorized. No dataset was downloaded, existing data/artifacts were preserved and no unapproved model was activated. Research paper, complete acceleration deployment, full hardware coverage, complete production verification and production infrastructure completion remain excluded.

| Area | Implemented and verified locally | Remaining evidence or work |
|---|---|---|
| Application | Camera management, authentication/roles, named detections, encrypted raw/titled sequence evidence, incident review, grouped role-routed alerts, acknowledgement and escalation | Representative real-camera interruption footage and operator usability evidence |
| Learning | Versioned imports, reviewed labels, split/hash/group checks, replay, calibration, lineage, acceptance/canary/rollback; on-demand per-class dataset coverage UI | Independent site labels and longitudinal retention evidence; automatic training remains disabled |
| Tracking | Class-aware global IoU assignment; offline MOTA/IDF1/per-class evaluator and synthetic correctness tests | Reviewed identity-labeled sequences, occlusion evaluation and real tracking measurements |
| Observability | Epoch losses, validation metrics, LR, elapsed time and assignment telemetry where recorded; selected checkpoint and stopping state; legacy and YOLO CSV compatibility | Historical fields absent from saved records remain unavailable |
| Model selection | Fingerprinted validation-only class rankings, contract grouping, native-profile deduplicated inference, approved-loaded fallbacks/cooldowns and tracker continuity; responsive comparison/progress dashboard | Independent pooled calibration, production approval and representative live-camera combined-route evidence |
| Models | Historical g4/g5 candidates; SVA v3 weapons completed with test AP50 0.0813; PPE v3 partial checkpoint preserved | PPE final evaluation unavailable; ten g6 jobs cancelled; every-class 0.50 target unmet; no superiority over YOLO established |
| Events/policies | Fall/fighting development candidates; configurable traffic, industrial, zone, PPE and temporal policies | Theft/collision supervision and trained candidates; reviewed signal/state/relation targets; independent fall/fighting footage |
| Hardware | CPU and laptop CUDA exercised historically; provider inventory and simulations; measured batch tuning | Full accelerator deployment and physical hardware coverage explicitly deferred |

The current pass verified 67 completed artifacts and evaluated/reused 57 calibrated matching-dataset validation reports. Actual two-expert weapons routing improved development F1 from 0.5444 (g4 alone) to 0.5969, but the newer three-expert route has higher recall/lower F1; component scores cannot guarantee pooled improvement. New optimizer/RNG checkpoints, gradient accumulation, partial lineage checks and YOLO letterbox parity are implemented. Full backend suite:212 passed, followed by 12 focused selection/curve/resource checks and five queue checks; Ruff, frontend build, authenticated desktop/mobile and generated evidence flow passed. No high-accuracy-all-classes or production-readiness claim is justified.

## Earlier g4 held-out development results

The newer weapons v3 test AP50 is 0.0813; PPE v3 has validation-only evidence. See [current training results](training-status.md). The table below is the g4 baseline snapshot.

AP50 measures detection ranking/localization, not percentage of correctly classified frames. Candidate scores do not establish site reliability or authorize activation.

| Family | Adapted YOLO AP50 | Scratch AP50 |
|---|---:|---:|
| Fire/smoke g4 | 0.5452 | 0.4058 |
| Dangerous items g4 | 0.5054 | 0.0315 |
| PPE g4 | 0.3223 | 0.0527 |

Fire scratch improved, while weapons/PPE retained identical parent weights after eight non-improving epochs. Fire stopped at epoch20 after best epoch12 plus patience8. See [g4 diagnosis](g4-sva-diagnosis.md) for per-class failure counts, data confounders and future controlled experiments. Fighting remains 14/20 clips correct with staged single-room weak labels; fall's ten all-correct test clips provide limited evidence. Neither proves general field performance.

The non-training suite passed **192 tests, with one real-training test deselected**. After adding YOLO CSV telemetry, **16 focused tests passed**, including the new test; repository Ruff and TypeScript/Vite build passed. Thus 193 distinct safe tests are covered across these runs, not one 193-test full-suite run. Existing dependency/deprecation warnings remain.

Browser checks verified the dataset coverage table against SH17's 17 classes and showed its three sparse evaluation warnings. Checkpoint/epoch controls were inspected in the running application. See [non-training completion record](nontraining-completion.md), [tracking format](tracking-evaluation.md), and `CONTINUE.md` for exact evidence and next steps.

## Architecture revision — 2026-10-13

SVA v3 is implemented with four-scale bidirectional features and dense quality-aware classification. Its weapons development run completed; PPE was interrupted, and other g6 jobs were cancelled. CUDA memory was profiled and PPE used batch 1. No v3 model is accepted or active. See [current results](training-status.md) and [v3 design and evidence](sva-v3-design.md).
