# SurveilX continuation checkpoint

Last updated: 2026-09-30. Read this file first after interruptions, then verify running processes and artifacts before resuming. This is a workspace progress record, not a research paper.

## User requirements

- Multi-domain system: parking, offices, warehouses, industrial facilities, retail and traffic.
- Add fire/smoke, firearms, fighting, falls, theft, traffic accidents/violations, PPE compliance and industrial hazards.
- Detection and video events first; no segmentation, pose or audio expansion unless needed and explicitly scoped.
- Custom scratch model plus modified YOLO; train/calibrate/evaluate later generations from generated, public and user-provided data.
- Ordinary computers supported; automatic workload selection; accelerator integrations validated separately.
- Use Python and native libraries; add custom C++ only for measured bottlenecks.
- Research paper excluded. Never equate implemented code, synthetic tests, downloaded weights or class names with validated real-domain capability.

## Completed implementation

- [x] Camera ingestion and generated streams, named detections, tracking, occupancy, encrypted evidence and review.
- [x] Authentication, roles, audit, hardware governor and global inference scheduling.
- [x] COCO/YOLO/VOC detection imports and entity-box video imports; four-way split and content/group guards.
- [x] Scratch SVA-Detector, adapted YOLO, pedestrian comparison and synthetic entity-event training.
- [x] Annotation/review, train-only version assembly, bounded replay, opt-in candidate training/calibration.
- [x] Frozen-model acceptance, lineage checks, artifact-bound approval, durable canary/rollback.
- [x] Task profiles and truthful `/capabilities`; completed/running/failed generation reporting.
- [x] Seven domain-scoped inherited 80-class candidates; named specialist deployment slots, domain routing, isolated tracking and rollback. At most eight slots per task, rotating one detection specialist per service.
- [x] Site-policy API/editor: object presence, restricted zone, configured wrong-way direction and possible missing helmet with visible/unambiguous head. Persistent review candidates, gap reset and cooldown; no full compliance claim.
- [x] Scene-level event importer and SVASceneNet training/calibration/runtime. Whole-video hash/group leakage checks, fixed-duration input contract, timestamped resampling and cadence abstention. No invented entity boxes. See `docs/scene-video-import.md`.
- [x] Scene-event incident policies: configured class, calibrated-model requirement, probability threshold, persistence, observation count, gap reset, cooldown and model-isolated state. Incidents name the detected event and preserve model/version evidence.
- [x] Crash-safe adapted-YOLO resume from a verified same-run checkpoint; lineage-checked scratch/event warm starts; old-domain retention reports for adaptation candidates. Independent acceptance remains mandatory.
- [x] Scene history memory bounded to 64 JPEG-compressed frames shared by event consumers; decode failures abstain safely.
- [x] Encrypted detection-sequence logging: bounded positive detections and policy/event incidents retain raw frames, titled review frames, raw/review clips and label metadata. The incident dialog shows a scrollable titled sequence; raw frames remain separate for reviewed annotation. Retention remains configurable (default seven days).

## Current model and data evidence

All listed new candidates remain inactive and unaccepted. Development metrics are not independent deployment validation.

| Family | Data and result | Remaining |
|---|---|---|
| General objects | Verified inherited YOLO11n 80-class taxonomy; seven domain-scoped candidates | Local calibration and independent domain evaluation |
| Person | Penn-Fudan scratch AP50 0.624879; adapted YOLO 0.864502; baseline 0.849291 | Broader independent evaluation; scratch superiority not achieved |
| Fire/smoke | D-Fire 672 images: 384 train, 96 each validation/calibration/test. `dfire-g1` completed: scratch AP50 0.088383; adapted YOLO 0.211201 | Accuracy inadequate; source identities unverified; larger audited data and independent acceptance |
| PPE objects | SH17 256 images, 17 source classes, 102 photographer groups; 163/29/25/39 splits. Scratch AP50 0.0135705. Crash-resumed adapted YOLO AP50 0.0476803, precision 0.566265, recall 0.129477; only person AP50 was useful | Accuracy inadequate; PPE-class AP50 values were zero; face_guard has zero test positives; scene/site independence unverified |
| Video events | Synthetic entity-motion models plus real UR Fall scene candidate. UR Fall has 40 original RGB clips, 10 per split; `urfall-g1` accuracy 0.50 on 10 test clips | Fall accuracy inadequate; fighting/theft/collision data, training and independent acceptance remain |
| Site policies | Four primitives implemented and tested | Real-camera validation; vest, proximity, speed/signals, machine-state and industrial hazards |
| Firearms | Verified 357,834,955-byte CC BY 4.0 Dangerous Items archive; 666-image bounded VOC development set. Scratch AP50 0.00978; adapted YOLO AP50 0.21298, precision 0.30769, recall 0.28346; firearm AP50 0.14558 | Accuracy inadequate; source identities unavailable; independent domain acceptance and activation remain |
| Accelerators | CPU execution; ONNX CPU/Azure installed; CUDA not validated; FPGA absent | Actual CUDA/TensorRT/FPGA/Jetson/Pi execution and benchmarks |

## Running processes and resume commands

### Accuracy campaign started 2026-09-30 (in progress)

- User requests better accuracy across every existing model family. Frozen plan: `data/generations/accuracy-g2/plan.json` (SHA256 `1dfcc7ec64093a7c25b076ea66d7426dba96bb54fd0109be8faf7603220854f9`). Twelve serial candidates cover fall, two synthetic entity-motion datasets, scratch/adapted detectors for SH17, Dangerous Items, D-Fire and Penn-Fudan, plus Penn-Fudan baseline YOLO. Obsolete predecessor runs are preserved.
- Launcher PID 1824; verify actual runner and child PIDs/command lines. Started with `.venv/Scripts/python.exe -m scripts.train_accuracy_campaign data/generations/accuracy-g2/plan.json`. Progress, child PID, logs and errors: `data/generations/accuracy-g2/status.json`. Before/after and per-class results: `comparison.json` / `comparison.md` in the same directory. Never start a second runner while this one or its child is alive.
- Resume the same command only after verifying no related process remains. The exclusive `runner.lock` contains the actual runner PID; remove a stale lock only after verification. Completed jobs are integrity checked and reused. Interrupted partial jobs are preserved and marked failed; inspect their logs and create a new version/plan for retries (this runner does not silently overwrite partial scratch/event runs).
- Implemented and focused-tested: adapted YOLO restores wrapped backbone and adapter weights before continuing; new YOLO training uses the runtime's square image geometry; scratch gets optional bounded class balancing; event continuation can fine-tune all layers with coherent clip augmentation; validation-based checkpoint selection includes the parent; event reports include class recall and confusion matrices. These are implementation improvements, not proof of improved accuracy until results finish.
- Recipe: detector continuations up to eight epochs at learning rate 0.0005, patience five; events up to 60 epochs at 0.0007, patience 15. Three CPU threads. The machine has no usable CUDA here. Calibration and test stay separate from training/selection; no model will activate automatically.
- Focused validation so far: 14 tests passed (accuracy regression tests, scene events and both existing generation runners). Full suite and final results remain pending.
- Scope limits: inherited 80-class domain copies have no local labeled full-taxonomy domain data; fighting/theft/collision and industrial hazards still lack trained real-data candidates. Rare/absent SH17 classes cannot be declared supported from an aggregate score.

Verify PID **and command line** with psutil before acting; Windows launcher and actual Python PIDs differ. Do not restart completed generations or remove a lock belonging to a live runner.

- Server: launcher PID 15060, actual PID 18816, unified exec session 86747; command `python -m uvicorn surveilx.api:app --host 127.0.0.1 --port 8000 --no-access-log`. `/health` returned 200 after the final build and tests. Verify command lines after any interruption because PIDs/sessions can become stale.
- Dangerous Items acquisition completed. Archive: `data/downloads/dangerous-items/dataset.zip`, 357,834,955 bytes, SHA256 `9c6749a3e36b6935fc468de9b1fb639b16c4caef2c21f5328a44ff210d3e1aa8`, 8,006 ZIP entries. The resumable downloader handles clean truncation and already-complete partial verification.
- Dangerous Items generation completed. Journal: `data/generations/dangerous-items-g1/status.json`. Scratch model ID `98d9edec-7b7d-461b-a22c-70599be79a82`; adapted model ID `a90a498a-55fe-49d5-968f-e73b15045fbd`. Both are inactive and deployment-ineligible.
- PPE generation is complete. Journal: `data/generations/sh17-g2/status.json`; adapted model ID `6db697d8-5956-4a73-bf3f-0c2cabb69674`.
- Correct SH17 manifest SHA256: `6b64e6a38baef86ba48c77a33af82fb0c381858bffd66c869a8b2bd46874c7ae`.
- D-Fire generation completed; `data/generations/dfire-g1/status.json`. Scratch ID `3602f4fa-a245-438a-9434-1d49b7c6c25c`; adapted ID `7ecfa4f1-d1e3-4fd4-879b-607f6334c351`.
- PPE scratch ID `662b09eb-8cba-4e8e-be70-63b4032764a0`; artifact `data/runs/sh17-g2-scratch/manifest.json`.
- UR Fall generation completed. Model ID `094c00ac-b631-4a9b-b10d-36f5f6cedd93`; artifacts `data/generations/urfall-g1/status.json` and `data/runs/urfall-g1-scene/manifest.json`.
- Existing real-source fallback HOG and demo synthetic fixture remain; no specialists activated during this batch.

## Acquisition provenance and known failures

- D-Fire archive: `data/downloads/dfire/dataset.zip`, 3,049,605,157 bytes, SHA256 `65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83`. Import rejects invalid boxes (340 source images), uses filename bands with omitted gaps, and explicitly marks source independence false. Acceptance rejects it as independent evidence.
- SH17: `scripts/prepare_sh17.py` + bounded HTTP ranges, strong ETag and ZIP CRC. 452,651,139 transferred bytes; full 14 GB archive not downloaded. Inventory `data/downloads/sh17/inventory.json`. Noncommercial development terms recorded. Glasses are not certified safety eyewear.
- SH17 README display order differed from authoritative numeric mapping in https://github.com/ahmadmughees/SH17dataset/blob/master/sh17.yaml . Initial `sh17-g1` immediately aborted (code 15); partial outputs preserved, unregistered. Corrected manifest records this. Class 10 helmet, 16 safety_vest. Only g2 is usable.
- UR Fall: importer acquired original cam0 RGB members through verified ZIP ranges, CRC and member hashes. Forty fixed-duration clips are complete in `data/datasets/urfall-development-v1`: five fall and five ADL sequences in each split. Sequence groups are disjoint; participant/site independence is unavailable. Publisher posture labels remain depth-frame metadata and were not copied as RGB frame labels. The model uses whole-clip sequence categories. Source is noncommercial academic CC BY-NC-SA 4.0.
- Dangerous Items: official Zenodo record 13786228, CC BY 4.0. The verified archive contains 4,009 annotations; 3,849 passed strict geometry/taxonomy checks. The bounded set has 666 images and 743 boxes: firearm 191, knife 200, machete 190, baseball bat 162; splits 399/96/65/106. `scripts/prepare_dangerous_items.py` maps Gun/Rifle to `firearm`, preserves the other labels, rejects 151 invalid annotations and 9 selected duplicate images, and discloses unavailable source identities. Development results cannot establish independent performance.
- A historical integrated test run collided with Vite replacing `frontend/dist/assets`. Avoid simultaneous frontend rebuild and API-import tests; the final sequential checks below pass.
- Earlier D-Fire filename parser and generation-panel JSX failures were fixed before completed imports/builds. Preserve data and all uncommitted work.

## Verification

- Full Python suite after logging and firearm generation changes: 80 passed, one Starlette/httpx deprecation warning, 110.81 seconds.
- Ruff passed across the repository. Frontend TypeScript/Vite production build passed.
- Focused UR Fall, acquisition, Dangerous Items and evidence-log tests pass. The 40-clip fall manifest and 666-image dangerous-item manifest validate.
- Live authenticated API verification passed after restart: 12 capability profiles, 10 experiments, 4 cameras and one configured policy on the first camera. Four bounded detection-sequence incidents were generated; the newest contained six raw/review frames, the review endpoint returned a valid titled JPEG and `reports/detection-sequence-preview.jpg` visually confirms `Detected: synthetic entity`. The in-app-browser tab reached the sign-in screen; an authenticated UI click-through remains pending. Prior screenshots are historical.
- Earlier reports remain historical: `reports/adaptation-verification.json`, `reports/acceptance-workflow.json`, `reports/multi-domain-verification.json`.

## Exact next steps

1. Acquire rights-compatible fighting, theft and traffic-collision event data. Preserve source groups and whole-video hashes, then train/calibrate/evaluate inactive candidates through `scripts.train_event_generation.py`.
2. Improve firearm/fall/PPE/fire-smoke accuracy with larger independently grouped data and multiple seeds. Current metrics are inadequate; do not activate these candidates.
3. Restart the API, verify `/health`, `/capabilities`, `/policies`, `/experiments` and the titled incident sequence in the browser, then refresh `reports/multi-domain-verification.json`.
4. Complete pooled-evidence calibration, drift triggers, learned environment inference and evaluated risk/energy optimization.
5. Validate actual RTSP failures, PostgreSQL/S3, distributed delivery/workers, CUDA/TensorRT/FPGA/Jetson/Pi execution, multi-domain rare-event recall and operator studies when required data/hardware are available.

Overall scope is incomplete. `docs/phase-status.md` tracks original phases. Research paper remains excluded.
