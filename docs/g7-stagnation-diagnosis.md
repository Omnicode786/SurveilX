# G7 accuracy diagnosis — 2026-10-15

Training, its retry queue and the automatic finalizer are stopped by user request. This pass used saved weights for inference, existing manifests/logs, code inspection and non-training tests. No new dataset, real training, calibration fit, model acceptance or activation was performed. All old artifacts are preserved. The corrections below are implemented but **not validated for accuracy**.

## Findings ranked by confidence and impact

| Finding | Evidence and confidence | Correction / remaining limitation |
|---|---|---|
| YOLO continuation bias warmup ignored the intended small LR | Certain configuration defect: requested LR 0.00005–0.0001, inherited bias warmup 0.1; epoch-one bias-group LR about 0.067. Causal share of the accuracy regression is unmeasured | New explicit fine-tuning policy starts all groups from zero, bounds warmup and records every group's LR range |
| Checkpoint selection and product evaluation used different metrics/paths | Certain: internal mAP50–95 versus final native AP50; multi-label NMS/default IoU 0.7 versus single-label/0.5; prepared JPEG versus original-image evaluation. PPE's best internal AP50 was epoch 9, best mAP95 epoch 12 | New fine-tuning selects checkpoints and stops on native original-image validation AP50; runtime geometry/decoder parity tested |
| Aggregate parent retention hid some genuine class gains | Certain on this validation set: saved candidate smoke, face guard and firearm AP50 improved despite overall regressions | Preserve trained candidates and expose fingerprinted class comparisons separately. They remain uncalibrated/inactive; no independent test gain established |
| SVA weapons classification/confidence is much weaker than oracle localization | Certain measured failure pattern; exact cause within supervision/optimization remains unresolved | Added class/foreground/background score and positive-IoU telemetry. Existing v3 already has a stronger decoupled class tower/four-level fusion; another untested tower expansion is not justified |
| PPE small-object detail and sparse/imbalanced classes constrain learning | Certain in the available labels: 17 classes, 631 train images, about 38:1 object-count imbalance; many objects only a few pixels wide after resizing | Milder augmentation and lossless preparation; tiny/sparse-class coverage remains a data constraint, not solvable by architecture alone |
| Repeated continuation uses limited development distributions | Certain data reuse; deployment generalization not measured | Preserve source-aware splits. No repeated epochs or larger model can establish unseen-site reliability without independent supervision |
| Original PPE decoding failure is consistent with host allocation pressure | Valid source/hash and later successful decoding are certain; allocation pressure is plausible, not proven | Bounded training decode, same-byte fallback and Windows commit checks; never skip an image silently |
| Optimizer resume could rewrite the prepared dataset before validation | Certain code defect found during review; not a proven cause of completed g7 results | Read-only integrity verification before resume; changed/legacy unverifiable recipes fail closed |

Evidence: `reports/g7-stagnation-audit.json`, `reports/g7-current-data-audit.json`, `reports/g7-diagnosis-weapons-v3.json`, `reports/g7-large-image-decode.json`, `reports/g7b-user-stop-diagnosis.json`. These reports are repository-relative paths; large artifacts remain local under `data/`.

## Why “nothing gained” was incomplete

G7's final aggregate gates retained stronger parents for the detector continuations. That avoided regressions, but reporting only those selected artifacts obscured trained class gains. A fresh read-only full-validation comparison uses identical geometry, RGB preprocessing, score floor 0.001, single-label NMS IoU 0.5, maximum 100 detections and the same AP evaluator for both checkpoints:

| Saved YOLO candidate | Parent overall AP50 | Candidate overall AP50 | Improved class examples: parent → candidate |
|---|---:|---:|---|
| Weapons, 448 letterbox | 0.569983 | 0.529100 | Firearm 0.535424 → 0.568160 (26 objects) |
| PPE, 512 stretch | 0.364622 | 0.356140 | Face guard 0.299334 → 0.392089 (15 objects); earmuffs 0.047136 → 0.055783 |
| Fire/smoke, 384 stretch | 0.414311 | 0.404241 | Smoke 0.368859 → 0.403734 (61 objects) |
| Person, 256 stretch | 0.978030 | 0.971055 | No class gain |

These are raw **validation** AP values. They do not establish calibrated operating precision/recall, independent test improvement or production readiness. The small last-digit difference from the original weapons candidate report comes from the audited direct-module versus predictor/fused inference path; the parent-retention conclusion is unchanged. Neither path rewrites weights.

The dashboard's new unpromoted-candidate panel is deliberately separate from calibrated model recommendations and runtime routes. Sidecars bind completed-manifest, candidate-weight and dataset hashes. A class-specialist candidate needs fresh calibration, a predefined evaluation/acceptance protocol and approved deployment before use. Repeated validation comparisons can overfit the development set, so these gains are hypotheses for future confirmation.

## Training-policy defect and selection mismatch

G7 explicitly chose AdamW with a small `lr0`. The installed Ultralytics trainer only resets `warmup_bias_lr` to zero in its automatic-optimizer branch; manual AdamW retained 0.1. The bias warmup was therefore 1,000× the requested rate for weapons/PPE, 1,333× for fire and 2,000× for person. Saved first-epoch `lr/pg2` values are 0.067033, 0.066911, 0.066795 and 0.068016 respectively. Those are actual epoch CSV entries; they are not inferred loss curves.

Inherited `nbs=64` with batch 4 also implied nominal accumulation 16 after warmup. On short campaigns, the three-epoch warmup and library's minimum iteration budget consumed a substantial fraction of optimization. These facts establish an unsuitable continuation recipe; only a controlled future experiment can measure how much accuracy they cost.

The new `finetune-v2` profile uses explicit AdamW, default LR 0.0001 when absent, zero bias warmup, momentum 0.9, nominal effective batch `max(8, batch)` unless explicitly configured, requested warmup at most one epoch, with the library minimum-iteration floor further capped at 10% of the total iteration budget. Scale is 0.15 and translate 0.05, with milder HSV changes. Mosaic remains disabled. Fresh automatic/pretrained legacy behavior remains available through `legacy-v1`; no old recipe or artifact is retroactively relabeled.

The new trainer runs native validation on the EMA weights every epoch and uses its AP50 consistently for best-fitness, checkpoint selection and early stopping. It preserves the library's additional metrics without comparing mAP95 and AP50 as if interchangeable. This adds validation inference time. Source references: [Ultralytics training arguments](https://docs.ultralytics.com/modes/train/), [configuration](https://docs.ultralytics.com/usage/cfg/) and the installed `ultralytics/engine/trainer.py`; local saved `fit/args.yaml` and `fit/results.csv` establish what actually ran. Versions without the verified warmup hook now fail explicitly for the new profile.

## SVA failure attribution and the fire control

The completed g6 v3 weapons weights were rechecked on all 96 validation images / 104 objects with frozen calibration. Every ground-truth object belongs to one exclusive category:

| Class | Not localized | Localized, wrong class | Correct class, low confidence | NMS/matching | Correct |
|---|---:|---:|---:|---:|---:|
| Firearm | 2 | 7 | 16 | 0 | 1 |
| Knife | 4 | 3 | 12 | 0 | 9 |
| Machete | 4 | 2 | 15 | 0 | 6 |
| Baseball bat | 1 | 13 | 6 | 0 | 3 |
| Total | 11 | 25 | 49 | 0 | 19 |

Raw-anchor localization is an **oracle upper bound**, not operational recall: it ignores classification and confidence and searches many anchors. Frozen operating recall is 19/104 and there are 78 false positives. Lowering the threshold or quoting 93/104 as recall would conceal the problem. Weapons examples in this corpus are mostly large and classes are fairly balanced; tiny-object assignment and imbalance are not the leading measured explanation there.

Fire's g4 scratch gain 0.237131→0.405813 coincided with 900 train-only additions, two classes and the focal recipe. It demonstrates learnability, but does not isolate a causal focal-loss effect. G7 reused that corpus and retained the same fire parent after four stale epochs. Weapons has just 399 train images / 443 boxes across four classes. PPE has many more semantic distinctions, tiny details and rare labels. There is no controlled evidence that focal loss alone helped fire and harmed the others. Existing v3 quality loss, localization and architecture were preserved; synthetic outputs/losses/gradients match the frozen implementation bitwise.

## Dataset and image-memory evidence

| Dataset | Train images | Classes | Empty train images | Recorded train groups | Largest source |
|---|---:|---:|---:|---:|---:|
| Dangerous items v1 | 399 | 4 | 0 | 399 | 0.6 MP |
| SH17 v2 | 631 | 17 | 0 | 205 | 32.669 MP |
| D-Fire v2 | 1,284 | 2 | 239 | 3 | 11 MP |

Recorded group identifiers describe the manifest's grouping granularity, not a verified number of independent sites. In particular, D-Fire's three groups must not be presented as only three unique scenes, and weapons' one-group-per-image cannot prove visual independence. Prior full manifest geometry/group analysis remains in `reports/g4-dataset-analysis.json`.

At a 512-pixel PPE canvas, median short-side sizes are ear 7.11 px, helmet 12.26 px, gloves 17.76 px and vest 22.48 px. Under-8-pixel counts are 323/577 ears, 84/261 helmets, 134/461 gloves and 38/233 vests. The old scale setting allowed further shrinking. Halved-size statistics in the report are configured-bound sensitivity, not observed augmentation frequencies. Medical suit has only 34 training objects in 27 images / 13 recorded groups and just two calibration objects; a new head cannot manufacture reliable calibration evidence.

SH17 includes 142 training sources above 12 MP. The failed valid 7000×4667 JPEG requires about 98 MB for its full BGR pixel buffer alone. Bounded native JPEG decoding produced 1750×1167 / 6.13 MB, retaining more than twice the 512 canvas along each axis. This is about 16× less pixel-buffer storage, not a measured 16× process-memory reduction. Single-image decode/resize timing changed from 1.028 to 0.786 seconds and mean pixel difference was 3.15/255; neither is an accuracy benchmark. Only training views use this optional reduction. Validation, calibration, test and runtime retain full decoding.

## Implementation, diagnostics and compatibility

- `training/yolo_policy.py`, the native trainer and lossless preparation fix the evidenced continuation/evaluation paths. New prepared-data fingerprints prevent geometry/source/policy changes during optimizer resume; legacy unverifiable runs require new-version weight initialization.
- `training/image_io.py` bounds training JPEG decode and retries the same file bytes if OpenCV returns no image. Failed decoding raises an explicit error; labels/images are never silently skipped or replaced.
- SVA telemetry records positive IoU, low-IoU positives, true-class foreground scores, maximum background scores, per-class assignments and source/decoded pixel-buffer bytes. Objective and gradients are unchanged.
- YOLO native histories record per-class metrics, AP50/P/R/F1, losses, all LR-group ranges, optimizer step attempts, physical/nominal batch, accumulation and elapsed/native-validation time. Attempts are not claimed to be successful updates when a scaler could skip. Old histories remain readable; missing historical values stay unavailable.
- The watcher, generation API and model dashboard read the new histories. Both pipelines preserve unpromoted trained candidates before aggregate parent selection. Current four saved YOLO candidates were evaluated without fitting and surfaced using sidecars; original manifests/checkpoints were not rewritten.
- Queue continuation archives stale resource errors in future journals and can check Windows commit headroom. The stopped g7 journal's historical error remains preserved.

Risks: new augmentation, reduced training decode and PNG preparation change future training pixels and disk/time requirements; empirical accuracy is unknown. Native validation adds work. Old checkpoint forward compatibility is retained, but exact optimizer resume is intentionally rejected when source/policy/assets cannot be verified. Frozen g7/g7b plans now have source mismatches and must not be reused. Source snapshots and all partial/completed weights remain intact. No candidate calibration or acceptance threshold was altered.

## Verification and smallest next experiment

The final verification record is `reports/g7-diagnosis-final-verification.json`. Full safe Python suite: **228 passed, 3 real-training tests deselected**, six existing dependency/ONNX warnings. The `--no-training` guard forbids training entry points and Adam/AdamW/SGD steps. Real saved-checkpoint inference parity, synthetic loss/gradient parity, resume-integrity rejection, callback/history/watcher formatting, routing isolation and bounded-decode geometry passed. Ruff and the TypeScript/Vite build passed; final authenticated API/browser and artifact checks are recorded separately. No synthetic check is treated as real-world accuracy.

If the user later authorizes training, first run **one matched weapons YOLO continuation**, from the same frozen parent, 448 letterbox, identical splits/seed and conservative physical batch. Use the corrected explicit bias warmup and native AP50 selection, retain all class metrics and compare against the existing g7 control at the same geometry. Use a new immutable source snapshot/output; do not resume an old plan. A small bounded run can test whether the severe continuation instability is reduced; it cannot promise a large gain or causal isolation of every policy component. Follow with a separate PPE data/scale experiment only if justified. Keep calibration/test independent and do not select on test. No runnable plan or queue was created in this pass.

Real rare-class/site data, independent temporal clips and genuine theft/collision/state/relation annotations remain prerequisites. Perfect accuracy is not a credible acceptance claim for this system.
