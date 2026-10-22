# Training status — 2026-10-15

**Capacity-tuned G8c is running under the user's renewed training authorization.** The first G8 job reached native validation but failed in metric recording because the installed trainer returns dictionary losses. The recorder now supports both dictionary and legacy tensor losses; 29 focused safe tests and lint passed. That failed trial and interrupted follow-on preparation are preserved. G8b was frozen but never launched.

## Current capacity-tuned G8c campaign

Plan SHA256: `47392b1c5095c56e48d7626df8ac9cf4c5d230b3dd871a212b045a471b479cb6`.

`data/generations/accuracy-g8c-capacity-1015/plan.json` creates nine new outputs with current frozen sources. Weapons/PPE/fire YOLO and weapons/PPE SVA have maximum 24 epochs and patience 8; legacy fire SVA 16/6, fighting 12/4, fall 4/2, person 6/2. These are ceilings: validation must justify continuing, and higher limits do not guarantee gains.

Real training-sample capacity probes on the MX130 (2 GiB), without saving changed weights, measured:

| Probe | Physical batch | Peak reserved VRAM | Training-step images/s | Outcome |
|---|---:|---:|---:|---|
| PPE YOLO, 512 px | 4 | 752 MiB | 9.28 | Fits |
| PPE YOLO, 512 px | 8 | 1,396 MiB | 10.29 | Selected; about 11% higher throughput |
| PPE YOLO, 512 px | 12 | — | — | Out of memory at probe budget |
| Weapons YOLO, 448 px | 8 | 1,074 MiB | 13.76 | Fits |
| PPE SVA v3, 512 px | 1 | 880 MiB | 1.14 | Fits |
| PPE SVA v3, 512 px | 2 | 1,686 MiB | 1.29 | Selected; about 14% higher throughput |
| Weapons SVA v3, 384 px | 4 | — | — | Out of memory; keep measured batch 2 |

These are short loss/backward/AdamW step probes on dense existing training samples, not end-to-end FPS, sustained utilization or accuracy benchmarks. Evidence: `reports/g8-capacity-probe.json` and `reports/g8-ppe-capacity-probe.json`. Weights were discarded after each probe; all parent artifacts remain untouched.

Four YOLO jobs use physical/effective batch 8, corrected `finetune-v2`, native original-image validation AP50 selection and zero bias warmup. PPE v3 uses batch 2/accumulation 2 (effective 4); weapons v3 keeps batch 2/accumulation 2. Other existing objectives/heads are preserved. Three CPU threads feed the GPU. The allocator budget is 85%, leaving memory headroom while compute is unrestricted. Utilization naturally varies during data loading and validation.

Jobs run serially with required CUDA and FP32. New-job guards require 1.8 GB disk, 3.5 GB physical RAM, 3.5 GB Windows commit headroom and 1,850 MiB free VRAM. Resource pressure pauses launches. The visible progress terminal and [dashboard](http://127.0.0.1:8000/performance) show current status. A CPU-only finalizer waits for campaign/training exit before refreshing validation recommendations. No model is automatically approved or activated.

Launch verification confirmed one CUDA trainer, sampled GPU94%/1325MiB, and a successfully saved first native epoch. Weapons validation AP50 is 0.566868 versus same-geometry parent 0.569983; no gain yet. All LR groups stayed within 0..0.000098, with 50 optimizer step attempts. Training continues; this is not a final test outcome. Evidence: campaign `launch-verification.json` and model `native_history.json`.

No dataset download/deletion. Theft, collision, traffic violations and industrial state/relation tasks still lack suitable existing supervision. Independent-site/event evidence and sparse PPE classes remain limitations. Earlier stopped instructions below describe old campaigns, not this new authorization. Sources of truth are current plan/status/log/history and `CONTINUE.md`.

## Historical stopped G7/G7b state

Main g7 finished with seven completed jobs and one failure. Its g7b retry was stopped after one completed epoch and is not being resumed or overwritten. All results below are historical; new G8 outcomes will be recorded separately.

## Final g7 results

AP50 values below are calibrated test metrics of the selected artifact. They are not ordinary classification accuracy. Checkpoint selection uses validation; test results never select models.

| Version | Completed epochs / maximum | Selected outcome |
|---|---:|---|
| `routing-g7-weapons-letterbox` | 12/12 | AP50 0.504067 → 0.514523; parent weights retained with new letterbox geometry/calibration |
| `routing-g7-fighting` | 6/8 | Validation fighting F1 0.631579 → 0.666667; test accuracy unchanged at 0.70 (14/20) |
| `routing-g7-ppe-sva-v3` | 0/8 | Failed decoding a valid 7000×4667 JPEG; partial files preserved |
| `routing-g7-ppe-yolo` | 12/12 | AP50 unchanged at 0.322344; parent retained |
| `routing-g7-fire-yolo` | 8/8 | AP50 unchanged at 0.540160; parent retained |
| `routing-g7-fire-scratch` | 4/8 | AP50 unchanged at 0.405813; parent retained, patience 4 |
| `routing-g7-fall` | 2/4 | Test accuracy unchanged at 1.0 on just ten clips; parent retained |
| `routing-g7-person` | 4/4 | AP50 unchanged at 0.939152; parent retained |
| `routing-g7b-ppe-sva-v3` | 1/8 | User interrupted during the following epoch. Completed-epoch validation AP50 0.122569 vs initialization 0.125835; no final calibration/test result |

Fighting completed **six**, not eight epochs: best epoch 2 plus patience 4. Earlier running snapshots were corrected. Fall's tiny perfect test and person scores do not establish general surveillance accuracy. Weapons letterbox changed class outcomes: firearm 0.527767→0.576910, knife 0.445310→0.346595, machete 0.434057→0.506119, bat 0.609135→0.628469. The selected weights equal their parent; these are preprocessing/calibration changes, not learned weight gains.

## Accuracy diagnosis and implemented corrections

The [detailed diagnosis](g7-stagnation-diagnosis.md) binds the saved schedules, data counts, full-validation comparisons and failure attribution. Main findings:

- Manual AdamW inherited bias warmup LR 0.1 despite requested LR 0.00005–0.0001. First-epoch bias-group LR was about 0.067. The new `finetune-v2` policy starts bias warmup at zero, bounds warmup to 10% of the iteration budget and defaults nominal effective batch to 8 instead of inherited 64. Exact accuracy attribution remains untested.
- YOLO selected its internal best by mAP50–95 using a different validation/decoder path from the product's final AP50 parent gate. New fine-tuning uses product-native, original-image AP50 for checkpoint selection and early stopping, with matched RGB, geometry, score floor and NMS.
- Aggregate parent retention hid some trained class gains. Fresh read-only validation found smoke AP50 0.368859→0.403734, face guard 0.299334→0.392089 and firearm 0.535424→0.568160. Overall candidate AP50 fell. These uncalibrated, unpromoted candidates are now visible separately in the performance dashboard; no test gain or activation is claimed.
- SVA weapons classification/confidence remains the main measured bottleneck. PPE also has tiny objects, 17 classes, about 38:1 object-count imbalance and sparse supervision. Architecture cannot supply absent examples.
- Bounded JPEG decoding reduces training pixel buffers for large sources while retaining at least twice the target resolution on both axes. Validation/runtime decoding remains unchanged. Lossless prepared images avoid a second JPEG compression step; milder scale/color augmentation avoids needlessly shrinking tiny objects.
- Resume preparation now verifies prepared images/labels, policy, source and geometry without rewriting them. Old unverifiable optimizer resumes fail closed; preserve their weights and use a new version for any future authorized continuation.

These are implemented source corrections, **unvalidated for accuracy until a future authorized experiment**. No post-fix real training has run. SVA architecture/forward/loss gradients and existing checkpoint pixels remain compatible; no unsupported capacity increase was made.

## Preservation, resources and future work

Both immutable plans and all old artifacts remain intact. Main plan SHA256: `6b9139bd644660fe8fe528763e45d9b374fd4a88d7d419b514bc04e61d0c48a2`; retry plan: `933aaab8028dd320547e28a480cb5d1adb4865dad8b902e7762b550cef8e3ad3`. Sources intentionally changed after stopping; the old frozen plans must fail source verification and must not be reused with this new recipe.

The saved retry `last.training.pt` contains optimizer/scheduler/RNG/sampler state; its exact checksum and stopped process identities are in `reports/g7b-user-stop-diagnosis.json`. Old g6 PPE has weights/history only. Never describe weight initialization from that run as optimizer resume.

Historical training used CUDA on the MX130 (2 GiB), FP32 and serial jobs. Sampled active batches reached 99%; allocator headroom does not cap compute utilization. The current API runs on CPU and automatic adaptation training is disabled. No training process remains. A read-only watcher may stay open and displays stopped/completed states.

No dataset, successful model, partial checkpoint or evidence was deleted. Earlier disk cleanup removed only pip's HTTP download cache. No new dataset was downloaded. The local server retains the process-only `SURVEILX_RETENTION_DAYS=36500` override to preserve pre-existing evidence; stored policy remains unchanged.

The smallest future experiment and compatibility risks are documented in the diagnosis. Independent site/actor data, sparse PPE examples, theft/collision/relation labels, pooled route calibration and real-camera acceptance remain incomplete. Perfect accuracy is not established or guaranteed. Research, full accelerator deployment and full production completion remain excluded.

## Historical G6 campaign

`cuda-accuracy-g6-v3-1013` stopped with **1 completed, 1 interrupted and 10 cancelled before starting**. The immutable plan SHA256 is `0ad27b595e4615b82673c8674776907802e37aab76a15c3fcbe8dd3665e29a10`.

| Model | State | Validation AP50 | Test AP50 |
|---|---|---|---|
| Weapons SVA v3 | Completed at epoch 30/48; best checkpoint epoch 14 | Parent 0.077472 → v3 0.149588 | Parent 0.031503 → v3 0.081309 |
| PPE SVA v3 | Interrupted after epoch 19/48 | Parent 0.087623 → best v3 0.125835, epoch 19 | Unavailable: calibration/test finalization did not run |
| Fire/person SVA v3, four adapted YOLO continuations, fighting/fall and two synthetic event checks | Ten jobs cancelled before starting | No g6 results | No g6 results |

Weapons stopped correctly after 16 epochs without improvement following best epoch 14. PPE was interrupted by request, not early stopping. Its saved selection state has no validation stopping reason. Existing g4/g5 results are not results for the cancelled g6 jobs.

AP50 is class-aware detection average precision at IoU 0.50, not frame classification accuracy. Compare validation with validation and test with test. The completed weapons test gain is 0.049806, or 4.98 percentage points. The 0.50 target remains unmet; no model was accepted or activated, and superiority over YOLO is not established.

### Weapons test results by class

| Class | Parent AP50 | SVA v3 AP50 |
|---|---:|---:|
| Firearm | 0.000548 | 0.024020 |
| Knife | 0.041948 | 0.047225 |
| Machete | 0.064074 | 0.140859 |
| Baseball bat | 0.019442 | 0.113132 |

## Earlier g5 outcome

G5 was stopped before the replacement campaign, preserving eight completed jobs and one partial PPE YOLO output. Its six scratch arms selected their frozen parents without improvement; weapons YOLO also retained its parent. Fire YOLO selected a candidate with improved validation AP50 (0.403295 to 0.414311), while test AP50 declined slightly (0.545198 to 0.540160). Test results did not select that checkpoint. These completed g5 artifacts remain preserved; they are distinct from the cancelled g6 continuations.

## Architecture and hardware

SVA v3 adds stride 4/8/16/32 features, compact weighted bidirectional fusion, a stronger decoupled classification tower, dense quality-aware class supervision and coverage assignment. It includes a six-epoch quality bootstrap and three-epoch learning-rate warmup. The default 17-class configuration has 1,619,004 parameters. This bundled experiment does not isolate which component produced the weapons gain. See [design and research](sva-v3-design.md).

CUDA was exercised on the NVIDIA MX130 (2 GiB), using the isolated `.venv-cuda` environment and serial jobs. Profiling reserved 820 MiB at 512 pixels/batch 1 and 1,000 MiB at 384 pixels/batch 2. PPE batch 2 at 512 pixels exceeded the profiling allocation budget, so its campaign used batch 1. These are sampled profiling measurements, not guaranteed whole-run maxima. CPU support remains available; full accelerator deployment is excluded. V3's four-level ONNX contract remains explicitly unsupported by the old three-level exporter.

## Preserved evidence and restart rules

Completed weapons weights match their manifest checksum. PPE preserves `weights.pt`, `history.json`, `config.json` and `dataset_hashes.json`, but has no completed manifest or calibrated test report. A partial SVA run has no verified optimizer-resume contract: a future authorized retry needs a new version and must preserve the interrupted output.

Local evidence includes `data/generations/cuda-accuracy-g6-v3-1013/status.json`, `status-before-user-stop-1014.json`, `runner.stopped-by-user-1014.lock`, both g6 run directories and `reports/g6-user-stop-1014.json`. Data and weights remain local under ignored `data/`; a Git clone does not contain these artifacts. No dataset was downloaded or model activated in this documentation pass.

Historical development verification before g6 passed 202 safe tests and 26 focused tests. The new pass has a full 212-test result plus subsequent focused checks, frontend build, authenticated desktop/mobile and live generated-evidence verification. See [measured selection outcomes](selection-results-1015.md) for current evidence.

## Remaining work

- Stronger validation and independently reviewed source/site data; the weapons gain is still low in absolute terms.
- Finish the authorized G7 queue and report PPE calibration/test results only after finalization; never report an interrupted checkpoint as a completed model.
- Sparse PPE classes and missing theft/collision/state/relation supervision require data, not merely more model capacity.
- Synthetic entity checks and person-only benchmarks do not establish real multi-domain surveillance reliability.
- Research paper, complete acceleration deployment, full hardware coverage and full production infrastructure/verification remain excluded.

The local web app is restored at localhost:8000. No learned production model is loaded; generated feeds use the explicit synthetic fixture and HOG remains the real-source fallback. The verification server uses process-only `SURVEILX_RETENTION_DAYS=36500` to preserve pre-existing evidence; `.env` and persistent policy were not changed. See [selection implementation](model-selection.md), [completion status](completion-status.md) and [continuation record](../CONTINUE.md).
