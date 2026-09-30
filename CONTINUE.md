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
| Firearms | Task profile, resumable acquisition of the 357,834,955-byte CC BY 4.0 Dangerous Items archive, and tested bounded VOC importer with canonical firearm/knife/machete/baseball-bat mapping | Complete and verify download; run importer and generation; evaluate and activate only after acceptance |
| Accelerators | CPU execution; ONNX CPU/Azure installed; CUDA not validated; FPGA absent | Actual CUDA/TensorRT/FPGA/Jetson/Pi execution and benchmarks |

## Running processes and resume commands

Verify PID **and command line** with psutil before acting; Windows launcher and actual Python PIDs differ. Do not restart completed generations or remove a lock belonging to a live runner.

- Server: actual PID 3220 in unified exec session 35522, `python -m uvicorn surveilx.api:app --host 127.0.0.1 --port 8000 --no-access-log`. `/health` returned 200 after the final build and tests. Verify the command line after any interruption because the PID/session can become stale.
- Dangerous Items acquisition is running in unified exec session 16857 with `python -m scripts.acquire_public dangerous-items --attempts 8`. The source repeatedly closes responses early, so the acquisition now resumes only after verified forward progress and allows eight bounded attempts per invocation. Status: `data/downloads/dangerous-items/acquisition.json`. Rerun the same command if all attempts are exhausted before 357,834,955 bytes.
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
- Dangerous Items: official Zenodo record 13786228, CC BY 4.0, archive expected size 357,834,955 bytes. The partial ZIP confirmed publisher VOC images/XMLs and source labels. `scripts/prepare_dangerous_items.py` is implemented and tested: Gun/Rifle map to `firearm`; Knife, Machete and Baseball Bat remain distinct; unknown/difficult/invalid annotations are rejected; content hashes prevent duplicates; and unavailable source identities are disclosed. Acquisition is incomplete until `dataset.zip` exists, inventory succeeds and SHA256 is recorded. Do not create or claim a firearm candidate from the partial archive.
- A historical integrated test run collided with Vite replacing `frontend/dist/assets`. Avoid simultaneous frontend rebuild and API-import tests; the final sequential checks below pass.
- Earlier D-Fire filename parser and generation-panel JSX failures were fixed before completed imports/builds. Preserve data and all uncommitted work.

## Verification

- Full Python suite: 75 passed, one Starlette/httpx deprecation warning, 111.29 seconds.
- Ruff passed across the repository. Frontend TypeScript/Vite production build passed.
- Focused UR Fall importer and event-generation tests: 2 passed. The 40-clip manifest validates and `urfall-g1` completed.
- Live authenticated API verification passed after restart: 12 capability profiles, 10 experiments, 4 cameras and one configured policy on the first camera. The new in-app-browser tab reached the sign-in screen; an authenticated post-restart visual pass remains pending. Prior screenshots are historical.
- Earlier reports remain historical: `reports/adaptation-verification.json`, `reports/acceptance-workflow.json`, `reports/multi-domain-verification.json`.

## Exact next steps

1. Let the Dangerous Items download finish; verify archive SHA256 and inventory, then run `python -m scripts.prepare_dangerous_items data/downloads/dangerous-items/dataset.zip data/datasets/dangerous-items-development-v1`. Audit the generated report, then run a new scratch/adapted detection generation. The importer keeps non-firearm weapon labels explicit rather than silently treating them as background.
2. Acquire rights-compatible fighting, theft and traffic-collision event data. Preserve source groups and whole-video hashes, then train/calibrate/evaluate inactive candidates through `scripts.train_event_generation.py`.
3. Improve fall/PPE/fire-smoke accuracy with larger independently grouped data and multiple seeds. Current metrics are inadequate; do not activate these candidates.
4. Restart the API, verify `/health`, `/capabilities`, `/policies`, `/experiments` and incident labels in the browser, then refresh `reports/multi-domain-verification.json`.
5. Complete pooled-evidence calibration, drift triggers, learned environment inference and evaluated risk/energy optimization.
6. Validate actual RTSP failures, PostgreSQL/S3, distributed delivery/workers, CUDA/TensorRT/FPGA/Jetson/Pi execution, multi-domain rare-event recall and operator studies when required data/hardware are available.

Overall scope is incomplete. `docs/phase-status.md` tracks original phases. Research paper remains excluded.
