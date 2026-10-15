# Training status — 2026-10-15

Training was explicitly stopped by the user on 2026-10-14. Rechecked on 2026-10-15: no campaign or model-training processes are running. Do not start, resume or queue training without renewed user authorization.

## Latest campaign

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

The last development verification before training passed 202 safe tests, with the actual scene-training test deselected; 26 focused tests and Ruff also passed. These are historical code checks, not a new production verification or accuracy test. This documentation update did not run training.

## Remaining work

- Stronger validation and independently reviewed source/site data; the weapons gain is still low in absolute terms.
- Final PPE calibration/test evaluation and future training only when separately authorized; do not report an interrupted checkpoint as a completed model.
- Sparse PPE classes and missing theft/collision/state/relation supervision require data, not merely more model capacity.
- Synthetic entity checks and person-only benchmarks do not establish real multi-domain surveillance reliability.
- Research paper, complete acceleration deployment, full hardware coverage and full production infrastructure/verification remain excluded.

The local web app was last observed offline on 2026-10-14; this is separate from saved artifacts and training state. See [completion status](completion-status.md) and [continuation record](../CONTINUE.md).
