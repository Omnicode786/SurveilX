# G4 SVA diagnosis and untrained improvements

Date: 2026-10-05. No real training, downloads, activation or acceptance occurred in this pass. AP50 is a detection metric, not percentage accuracy. Existing g2/g3/g4 artifacts remain frozen.

## Completed results and stopping

| Candidate | Parent test AP50 | Selected test AP50 | Best epoch | Completed / maximum | Reason |
|---|---:|---:|---:|---:|---|
| Fire YOLO | 0.478528 | 0.545198 | 12 | 20 / 30 | Eight epochs without improved Ultralytics AP50:95 fitness |
| Fire SVA | 0.237131 | 0.405813 | 12 | 20 / 30 | Eight epochs without improved validation selection |
| Weapons SVA | 0.031503 | 0.031503 | 0 (parent) | 8 / 30 | Eight epochs without beating parent |
| PPE SVA | 0.052669 | 0.052669 | 0 (parent) | 8 / 30 | Eight epochs without beating parent |

Scratch selection is lexicographic (validation AP50, negative reference BCE validation loss), including the parent at epoch zero. Weapons parent validation AP50 is 0.077472; the best g4 epoch was below it. PPE parent validation AP50 is 0.087623; the best g4 epoch reached only 0.086630. Fire improved validation AP50 from 0.224226 to 0.355161. Tensor-by-tensor comparison proves that weapons and PPE selected the identical parent parameters. Flat exported results therefore do not mean that all training updates were zero. Rejected epoch weights are unavailable; they must not be inferred from the retained parent.

Stopping behaved correctly under the configured policy, with no off-by-one error. Eight stale epochs may be too restrictive for some future recipe changes, but that is an experiment-design question, not evidence of a broken implementation. YOLO uses a different selection metric: its best AP50 epoch was 17, while AP50:95 fitness peaked at 12. The saved process log explicitly reports patience eight and best epoch 12.

## Why the domains diverged

Fire received 900 additional training images: 384 to 1,284, with objects increasing from 628 to 2,065. The original records and held-out splits were preserved. Weapons and PPE received no corresponding expansion. Fire also has only two classes, moderate imbalance (1.61x), and almost complete positive assignment. Thus fire is evidence that SVA can improve, but not a controlled focal-loss experiment: data, batch size, device and loss recipe changed together.

On the same 133 validation objects, fire's correct operational detections rose from 31 to 47; confidence failures fell from 53 to 37, while raw localization failures only fell from 43 to 41. Forward classification BCE fell from 0.3492 to 0.2397, objectness loss from 0.5938 to 0.5011, and GIoU loss from 0.4176 to 0.3777. These are current frozen-checkpoint snapshots, not reconstructed epoch telemetry. Calibrations differ between parent and candidate, so operating-point counts reflect both the selected model and its frozen calibration. Better ranking/score separation and some geometric improvement are supported; a single causal explanation is not.

Weapons has 399 training images, 443 objects, four fairly balanced classes (1.24x), and no lost training objects under the historical assignment. Median short sides are 84â€“99 pixels at 384 resolution, with no short side below eight pixels. Elongation exists, but severe tiny-object scarcity or assignment starvation is not the leading explanation in this corpus. Of 104 validation objects, 74 have some raw localization at IoU >=0.5; 28 lack any correctly classified localized candidate and another 35 fail the score threshold. Only nine are operationally correct. Classification/confidence is the largest combined bottleneck, with localization still contributing 30 misses. Low score can involve objectness as well as classification; it must not all be attributed to the class tower.

PPE combines 17 classes, crowded overlapping labels, small body/PPE parts and severe imbalance. Among 8,062 training objects, 876 (10.87%) receive no positive location because the smallest-area conflict rule lets other objects take every eligible location. Head loses 311, person 187, tools 97, helmet 43 and safety vest 48 objects. This is a demonstrated supervision defect. PPE's validation errors are mixed: 672 raw localization failures, 332 wrong-class failures and 570 confidence failures out of 1,893 objects. Classification alone will not solve it.

PPE data scarcity is separate: medical suit has only 34 objects in 27 training images and two calibration objects. Face guard and safety suit have only six test objects each. Class instance imbalance is 38.38x. Ear median short side is 7.11 pixels; 323 ear objects are below eight pixels. Tools and helmets also contain many small instances. Repeated sampling cannot create missing independent examples, and a larger head cannot make these class estimates reliable.

## Ranked causes and confidence

1. **High confidence: retained-parent selection explains unchanged exported weapons/PPE scores.** Actual parameters match, histories fail the selection threshold, patience is exhausted correctly.
2. **High confidence: PPE positive-assignment starvation and class/data sparsity.** Measured from every manifest label. Fixing assignment supplies supervision but does not establish accuracy or solve missing data.
3. **High confidence: weapons classification and confidence bottleneck.** Full validation decomposition supports it. **Moderate confidence:** limited classifier capacity is a contributing cause; only a controlled future comparison can establish this.
4. **High confidence: fire's substantially larger training set and simpler class task distinguish it.** Contribution sizes cannot be disentangled from the other recipe changes.
5. **Moderate confidence: focal recipe mismatch and scale changes deserve isolation.** Classification focal loss runs only on positive-object locations, while objectness already uses dense focal loss. At logit zero, gamma 2 / alpha .25 scales true-class absolute BCE gradients by about .149 and other-class gradients by .447. It changes pressure as well as loss magnitude. This differs from the dense-background motivation of the original [focal loss paper](https://arxiv.org/abs/1708.02002). Existing artifacts do not prove that focal loss helped fire or harmed the others; Adam also prevents interpreting gradient scaling as a direct step-size ratio.
6. **Moderate confidence for PPE; lower for weapons: input scale and shape distortion.** Square resize changes native aspect ratios. Letterbox preserves shape but makes many small objects smaller on the same canvas; it is not an automatic improvement. Existing training already applies horizontal flips and bounded brightness changes.
7. **Low confidence as primary cause: duplicate/NMS losses.** They account for only 2/104 weapons and 38/1,893 PPE objects under the diagnostic definition.

Source groups are proxies, not proof of independent filming sources: D-Fire has three recorded training families, weapons 399 image groups, PPE 205 groups. The dataset report measures source repetition and exact content redundancy, but cannot establish semantic near-duplicates or unseen field generalization from these identifiers.

## Implemented changes; accuracy unvalidated

- Optional `coverage-v2` assignment reserves one eligible location per object through maximum bipartite matching, then retains the existing assignments elsewhere. It preserves the location set, scale rule, center radius, box coder and regression targets. Labels-only results: D-Fire v1 lost objects 4â†’0; v2 11â†’0; weapons 0â†’0; PPE 876â†’0. Over-capacity synthetic cases remain explicitly observable. This has CPU/GPU synchronization overhead that needs future profiling.
- Optional SVA architecture version 2 adds a residual depthwise/channel classification refinement, initialized to exact identity. Only 19,680 parameters are added, roughly 1.4% over the 1.40M model. Existing FPN, backbone, objectness and localization are preserved. An explicit upgrade accepts exact v1 state keys/shapes; actual frozen checkpoints give identical initial outputs. No upgraded weights were persisted or activated.
- Explicit classification-only adaptation freezes shared features and geometry, allowing a later focused head experiment. GroupNorm has no running-stat updates. This option requires a trained parent.
- Shared optional letterbox preprocessing transforms images, boxes and zone maps and inversely maps predictions for evaluation/runtime. Historical stretch remains the default. Continuation rejects silently changing parent geometry. Runtime, acceptance and profiling paths carry artifact preprocessing metadata.
- Clear validation selection helper preserves the historical policy and reports parent selection, best epoch, stale count, criterion and stopping reason. Atomic history writes reduce partial-read risk.

No FPN rewrite, extra scale head, localization redesign or default focal change was made: the evidence does not justify bundling those changes now.

## Diagnostics and observability

Future scratch histories record total/objectness/classification/GIoU/smooth-L1 losses; validation AP50, precision, recall, F1 and per-class AP50; learning rate; sampled-image/positive counts; positives per FPN level; assigned/unassigned objects; elapsed time; selection state and recipe. Loss aggregation is explicitly mean batch objectives. Validation selection keeps its common reference BCE loss instead of silently using the training focal objective. `watch_training.py` displays these fields, and generation API results expose the latest epoch. Old histories remain readable and missing fields display as not recorded. No historical values were fabricated or training rerun to populate them.

The full-validation failure diagnostic uses existing weights and calibration, separates five mutually exclusive GT outcomes, and records per-class confidence, confusion, forward losses and assignment counts. Raw-anchor localization is an oracle upper bound, not operational recall. The decoder category includes top-k truncation, NMS, final cap and one-to-one GT competition; it is not a claim that every such failure is caused by NMS. This historical diagnostic explicitly rejects letterbox artifacts until its raw-anchor coordinate comparison is extended.

## Smallest future experiment â€” not launched or queued

Use the existing weapons parent and unchanged 399-image training/96-image validation split for two short, fixed-seed classification-only arms: v1 versus identity-upgraded v2. Keep stretch 384, center-v1 assignment, batch, sampler, optimizer, learning rate, BCE class-sum recipe and maximum eight-epoch budget identical. Use validation AP50 plus per-class diagnostics; retain the parent if neither improves. This isolates classifier capacity under BCE; it does not establish the causal effect of focal loss. Do not select from the test split. Profile the optional assignment separately, then run a separate PPE assignment-only comparison if authorized later. Keep fire as a regression control. No runnable generation plan was created.

## Evidence and compatibility

Reports in `reports/g4-dataset-analysis.{json,md}`, `g4-failures-*.json`, `g4-revision-verification.json`, `g4-augmentation-sensitivity.json` and `g4-cleanup-journal.json` contain exact evidence. The 12-image/domain forward augmentation sample is descriptive only: original/flip/brightened correct counts were fire 4/3/3, weapons 1/0/1, PPE 15/18/14. It does not identify augmentation as the cause of failed learning.

Default v1 state keys, config and stretch geometry are retained. V2 weights require the new loader; older code cannot load them. Future history consumers must tolerate additional fields; detection-loss telemetry now includes a per-level tensor, so consumers must select scalar loss keys. Optional letterbox requires matching artifact/runtime support, and v2 assignment performance on the laptop GPU remains unmeasured. No post-change accuracy claim is made. Broad multi-domain reliability, video-event gaps and production/hardware exclusions in CONTINUE.md remain unchanged.


## Full validation per-class failure counts

Order: not localized, wrong class, low confidence, decoder/matching, correct. Counts are at each frozen operating threshold, not AP.

### accuracy-g3-fire-scratch

| Class | GT | Not localized | Wrong class | Low confidence | Decoder/matching | Correct |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 61 | 17 | 1 | 36 | 0 | 7 |
| fire | 72 | 26 | 5 | 17 | 0 | 24 |

### cuda-g4-fire-scratch

| Class | GT | Not localized | Wrong class | Low confidence | Decoder/matching | Correct |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 61 | 13 | 0 | 29 | 1 | 18 |
| fire | 72 | 28 | 5 | 8 | 2 | 29 |

### cuda-g4-weapons-scratch

| Class | GT | Not localized | Wrong class | Low confidence | Decoder/matching | Correct |
|---|---:|---:|---:|---:|---:|---:|
| firearm | 26 | 8 | 11 | 4 | 0 | 3 |
| knife | 28 | 8 | 6 | 9 | 0 | 5 |
| machete | 27 | 7 | 2 | 18 | 0 | 0 |
| baseball_bat | 23 | 7 | 9 | 4 | 2 | 1 |

### cuda-g4-ppe-scratch

| Class | GT | Not localized | Wrong class | Low confidence | Decoder/matching | Correct |
|---|---:|---:|---:|---:|---:|---:|
| person | 316 | 61 | 37 | 118 | 11 | 89 |
| ear | 129 | 94 | 5 | 13 | 3 | 14 |
| earmuffs | 30 | 5 | 17 | 7 | 1 | 0 |
| face | 176 | 47 | 28 | 37 | 9 | 55 |
| face_guard | 15 | 1 | 10 | 4 | 0 | 0 |
| face_mask | 34 | 6 | 7 | 12 | 0 | 9 |
| foot | 33 | 19 | 7 | 6 | 0 | 1 |
| tools | 59 | 16 | 19 | 24 | 0 | 0 |
| glasses | 47 | 23 | 17 | 6 | 0 | 1 |
| gloves | 114 | 51 | 25 | 38 | 0 | 0 |
| helmet | 96 | 38 | 32 | 23 | 0 | 3 |
| hands | 279 | 113 | 30 | 94 | 6 | 36 |
| head | 299 | 80 | 37 | 112 | 5 | 65 |
| medical_suit | 34 | 2 | 10 | 22 | 0 | 0 |
| shoes | 148 | 99 | 7 | 31 | 3 | 8 |
| safety_suit | 28 | 3 | 16 | 9 | 0 | 0 |
| safety_vest | 56 | 14 | 28 | 14 | 0 | 0 |


## Non-training verification results

- Full safe suite: 178 passed, two synthetic disk-reserve fixture failures, one real-training test deliberately deselected. Both failures were isolated to host free space, not model code. Fixtures now supply deterministic free space; the real acquisition reserve remains unchanged and the low-space rejection is also tested.
- Follow-up: all 12 affected/diagnostic tests passed; final revision/geometry/watcher checks passed 10/10 after adding flip parity. Across the suite and targeted reruns, all 183 selected non-training tests passed. The excluded test is `test_scene_training_calibration_runtime_and_cadence`.
- `--no-training` blocks trainer entry points and Adam/AdamW/SGD optimizer steps. The allowed synthetic backward test performs no optimizer update. Dataset download tests use in-memory fake responses, not internet acquisition.
- Repository Ruff passed. Actual old checkpoint strict loads and v2 identity outputs passed for all three domains. Labels-only assignment verification covered every training object in all four manifests. ONNX/export, backend and decoder checks in the safe suite passed with existing deprecation/tracer warnings. No frontend files changed; no frontend build was needed.
- Read-only watcher completed successfully. Product readiness verifies 56 completed model weights, zero errors, and preserved interrupted originals/replacements. Frozen integrity snapshot verified 275 files unchanged.


## Completed storage cleanup

Removed 15 verified derived YOLO caches, recovering 1,054,997,989 bytes (about 1.055 GB). All canonical dataset versions, archives and frozen model artifacts remain intact. The older D-Fire g1 cache is retained because a validation image was not byte-reproducible under the available source/recipe; the failed initial verification deleted nothing. The journal records original-file copy versus resized-image reconstruction per image, exact YAML, labels and hashes. Final verification confirms 275/275 frozen files unchanged, no training processes, and 5,270,790,144 bytes free.
