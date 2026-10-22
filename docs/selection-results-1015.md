# Selection and accuracy evidence — 2026-10-15

SurveilX now compares compatible models by category and routes explicitly loaded experts accordingly. **Recommendations remain inactive.** This pass does not establish production readiness, high accuracy for every class or scratch superiority over YOLO. See [implementation and commands](model-selection.md).

## Architecture review and decisions

| Family | Evidence-backed decision |
|---|---|
| Legacy SVA detection v1/v2 | Retain the successful fire focal recipe. Prior g4 diagnostics found strong weapons localization with weak class discrimination; repeated classifier-only/assignment arms did not improve retained checkpoints. Avoid another identical weapons-scratch retry |
| SVA detection v3 | Existing four-scale/bidirectional features, stronger classification tower and dense quality-aware supervision improved weapons test AP50 0.0315 → 0.0813 but remain poor. Continue the stronger preserved PPE partial rather than discard its learned weights. Fix bootstrap reset, effective batch and resume infrastructure; no unmeasured capacity inflation |
| Adapted YOLO | Strongest current detection family for most categories. Test letterbox on elongated weapons, preserve g4 knife and task-specific PPE/fire comparators, use exact training/runtime BGR and geometry parity |
| SVA-Net scene/entity events | Keep scene pixels separate from entity-only trajectories and detection metrics. Add reproducible stateful checkpoints, explicit per-class/calibration diagnostics and a bounded fighting continuation; do not turn weak inherited clip labels into precise event-onset claims |

V3's prior design draws on [GFL](https://arxiv.org/abs/2006.04388) and efficient feature fusion; it is documented in [SVA v3 design](sva-v3-design.md). [Class-balanced loss](https://arxiv.org/abs/1901.05555) was reviewed, but stronger reweighting was not added without evidence: sampling cannot create missing examples. Checkpoint/scheduler infrastructure follows the [PyTorch optimizer contract](https://docs.pytorch.org/docs/stable/optim.html). No YOLO internals were blindly copied into SVA.

## Comparable model inventory

At this snapshot, **67 completed artifacts** passed checkpoint checks; **57 calibrated, matching-dataset validation reports** are available, covering 47 task/dataset/category groups. Ten uncalibrated/unmatched artifacts are visible but ineligible. Partial/interrupted outputs are listed separately. Older equivalent weights reuse their verified evaluation; counts do not imply 57 distinct architectures.

Each sidecar records weight/dataset/model-manifest hashes and evaluation split. Results include per-class AP50, AP50–95, P/R/F1 and FP/FN for detection; event confusion matrices, per-class F1/missed-event rate, ECE/Brier/NLL. Detection calibration scores describe retained predictions; missed objects are measured separately by recall. Historical fields that were never saved remain unavailable.

## Current category recommendations

These are validation-only choices on the named development corpus, not interchangeable global rankings. Multiple versions can serve different classes in the same corpus. The full snapshot and all fallback scores are in `reports/current-model-recommendations.json`; the live dashboard refreshes from completed artifacts/evaluations.

| Category / corpus | Recommended version | Validation AP50 | Recall | F1 |
|---|---|---:|---:|---:|
| Firearm / dangerous-items v1 | `accuracy-g3-weapons-yolo` | 0.5780 | 0.6538 | 0.5763 |
| Knife / dangerous-items v1 | `cuda-g4-weapons-yolo` | 0.7236 | 0.6429 | 0.6923 |
| Machete / dangerous-items v1 | `routing-g7-weapons-letterbox` | 0.4600 | 0.4444 | 0.4364 |
| Baseball bat / dangerous-items v1 | `routing-g7-weapons-letterbox` | 0.6192 | 0.4783 | 0.6111 |
| Fire / D-Fire v2 | `cuda-g4-fire-yolo` | 0.4643 | 0.5139 | 0.5034 |
| Smoke / D-Fire v2 | `cuda-g5-fire-yolo` | 0.3689 | 0.3934 | 0.4615 |
| Helmet / SH17 v2 | `cuda-g4-ppe-yolo` | 0.5246 | 0.4063 | 0.5379 |
| Safety vest / SH17 v2 | `cuda-g4-ppe-yolo` | 0.2720 | 0.3214 | 0.3396 |
| Gloves / SH17 v2 | `cuda-g4-ppe-yolo` | 0.1285 | 0.1404 | 0.1928 |
| Face mask / SH17 v2 | `cuda-g4-ppe-yolo` | 0.6662 | 0.5294 | 0.6316 |
| Person / SH17 v2 | `accuracy-ppe-g3-yolo` | 0.8110 | 0.7816 | 0.8192 |
| Person / Penn-Fudan | `accuracy-g2-person-baseline` | 0.9780 | 0.9322 | 0.8871 |
| Fighting / AIRTLab | `routing-g7-fighting` | Not a detection task | 0.6000 | 0.6667 |
| Fall / UR Fall | `accuracy-g2-fall` | Not a detection task | 1.0000 | 1.0000 |

Fall has only five positive validation clips and five positive test clips. Fighting validation has ten positive clips, with shared staged room/actors across source splits. These are weak reliability estimates. PPE tools/feet/earmuffs and several rare classes remain very poor; a class name or a winning score is not validated support. Theft, collisions and learned industrial state/relation hazards have no newly trained candidate. Existing configured policies remain distinct from learned event recognition.

## Executed combined-router comparison

The actual runtime router was exercised on all 96 dangerous-items validation images/104 objects, with native geometry and each model's fixed calibration threshold. No test selection, activation or threshold reduction was used.

| Configuration | Precision | Recall | F1 | False negatives | False positives |
|---|---:|---:|---:|---:|---:|
| g4 YOLO alone | 0.6447 | 0.4712 | 0.5444 | 55 | 27 |
| g3 YOLO alone | 0.6064 | 0.5481 | 0.5758 | 47 | 37 |
| g3 firearm/machete/bat + g4 knife | 0.6552 | 0.5481 | 0.5969 | 47 | 30 |

The two-model route improves F1 by 0.0524 and recall by 0.0769 versus g4 alone. Relative to g3 alone, it retains recall while reducing false positives from 37 to 30. Route selection and measurement share validation, so these are development gains, not independent acceptance. Component calibration does not establish pooled calibration. The new g7 letterbox route is evaluated separately rather than replacing this comparison.

The follow-up **g3 firearm + g4 knife + g7 letterbox machete/bat** route scored precision **0.5918**, recall **0.5577**, F1 **0.5743**, **46 FN/40 FP** on the same 96 images. It misses one fewer object than the two-model route but adds ten false positives and costs a third inference. It therefore does **not** establish a better combined detector. Transparent per-class AP/recall weighting can recommend a component while pooled F1 regresses; both are shown, and neither route is deployment-approved. The two-model configuration is the stronger measured F1 option. A deployment review must assess the desired missed-event/false-alarm tradeoff and validate the actual combination independently. Evidence: `data/routing-evaluations/dangerous-items-routing-g7-v1.json`.

## Warm native runtime measurements

MX130 CUDA, two warmups/eight validation images, median single-image adapter time including preprocessing/forward/decode/calibration; excludes file/camera I/O. Hardware and sample count are stored with each measurement. These are per-model costs, not end-to-end multi-camera FPS or energy measurements.

| Version | Median ms |
|---|---:|
| `accuracy-g2-person-baseline` | 62.37 |
| `accuracy-g3-weapons-yolo` | 64.56 |
| `cuda-g4-weapons-yolo` | 72.33 |
| `cuda-g4-ppe-yolo` | 77.74 |
| `cuda-g4-fire-yolo` | 66.95 |
| `cuda-g5-fire-yolo` | 57.01 |
| `cuda-g6-weapons-sva-v3` | 123.08 |

SVA v3 is slower and less accurate on weapons here; no superiority claim is justified. No custom C++ rewrite is warranted from these data because PyTorch/OpenCV already execute the heavy kernels natively.

Controlled original/dark-gamma/blur probes were also executed on existing labeled validation images. Sampling now covers available classes before filling (36 images/model), records exact sample names and support, and includes an empty image when available. Transforms do not create unseen sites or genuinely reviewed night/blur footage; truncated-threshold AP is deliberately not reported. These probes are not used for selection.

| Fixed-threshold probe F1, 36 images | Original | Dark gamma 1.8 | Mild blur |
|---|---:|---:|---:|
| g4 weapons YOLO | 0.5714 | 0.5246 | 0.5714 |
| g4 PPE YOLO | 0.5834 | 0.5574 | 0.5751 |
| g5 fire/smoke YOLO | 0.5667 | 0.5893 | 0.5920 |

All four weapon classes and all 17 SH17 classes have support in these bounded probes. Fire's transformed-sample gain does not establish general improvement under real darkness/blur. No geometry label was altered and no synthetic metric was reported as real-world accuracy.

## Training outcomes and runtime state

Main g7 finished seven jobs and one PPE decode failure. The user stopped g7b after one completed epoch; all training/queue/finalizer processes are stopped. Most final artifacts retained parents, while fresh saved-candidate validation exposed smoke, face-guard and firearm gains hidden by aggregate selection. Those candidates remain uncalibrated and inactive. Weapons preprocessing gains are mixed, including knife regression; fighting test accuracy remains unchanged. See [training status](training-status.md) and [diagnosis](g7-stagnation-diagnosis.md) for final counts, evidence and implemented but accuracy-unvalidated fixes. Earlier g6 artifacts remain preserved.

**No learned model is loaded in the local runtime.** Existing enabled cameras use generated sources and the explicit synthetic-color fixture; HOG is available for real sources. The new model-selection code is integrated with normal expert deployment and tested with actual saved-model CPU inference, but no unapproved candidate was deployed to demonstrate it.

## Verification and limits

- Full Python suite: 212 passed, six existing dependency/ONNX warnings; subsequent selection/curve/resource suite:12 passed; retry queue:five passed. One overlapping focused run hit an Argon2 memory-allocation error; serialized retry passed that case. A new curve fixture initially lacked its JSON import; corrected without changing product behavior.
- Repository Ruff, TypeScript/Vite build and authenticated desktop/mobile dashboard verification passed. Mobile had no document overflow and browser recorded no page errors.
- A newly created generated camera advanced frames and returned live JPEGs, encrypted evidence, titled review frames and WebSocket updates; only its new incident was acknowledged. The test camera was disabled afterward; pre-existing cameras/incidents were preserved. Evidence: `reports/routing-live-verification.json`.
- No dataset download, model acceptance/promotion or artifact deletion. Only reconstructible pip download cache was purged to restore disk headroom.

Real-camera occlusion/movement/darkness reliability, identity tracking scores, camera-hour false alarms, critical-event delay and independent multi-site acceptance are still unavailable. No deployment threshold was weakened. Hardware deployment/coverage, full production infrastructure/verification and the research paper remain excluded.
