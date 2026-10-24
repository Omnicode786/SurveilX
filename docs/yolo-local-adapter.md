# Modified YOLO continuation: local adapter experiment

This development experiment preserves YOLO11n and the existing scene/zone adapters. It does not claim a new independent detector, guaranteed accuracy improvement, or deployment readiness.

## Evidence and parent choice

The G9 fire control had native validation AP50 **0.425991**, versus **0.421102** for the heavier v3 recipe. The new campaign therefore uses the simpler v2 optimization recipe without v3's stem freeze or mosaic. This is an observed validation comparison of bundled recipes, not proof that any individual v3 setting caused the difference.

At identical 512 px letterbox geometry on the 96-image weapons validation split:

| Preserved weights | Native validation AP50 |
|---|---:|
| G3 completed | 0.540612 |
| G8c completed | 0.543506 |
| G9 interrupted best | 0.480944 |

G8c is the weapons initialization. G9 partial weights and optimizer state remain preserved. Training continues in new version directories with fresh optimizers; it does not claim an exact optimizer resume from G9.

Existing training labels show 399 weapons images and broadly balanced classes (97–120 objects per class). PPE has 631 training images, 17 classes and 34–1,305 objects per class. At 640 px letterbox, 320/577 ear labels and 77/261 helmet labels have a short side below eight pixels. These are dataset/resolution limitations; architectural changes cannot manufacture supervision or recover detail absent from source pixels. Counts use training only. Recorded source groups do not establish independent sites.

Detailed hashes, per-class counts, scales and validation results are in `reports/yolo-g10-design-review.json`.

## Versioned architecture

The original adapter combines globally pooled channel gates, one spatial mask shared across channels, and scene/zone modulation. The experimental `local-v2` adapter adds a 3×3 depthwise convolution whose bounded output independently refines each channel at each location:

`new(x) = old(x) + 0.1 * x * tanh(depthwise3x3(x))`

All new convolution weights start at zero. Every original backbone, head, adapter and context tensor is copied, so an upgrade initially reproduces the supplied detector exactly. The new branch adds **4,608 parameters** across the three adapter blocks; no extra detection head or stride change is introduced. The local correction is bounded by `0.1 * abs(x)` and has a nonzero gradient at initialization. Old checkpoints keep the original adapter class and remain loadable. Downgrading a trained local adapter by discarding its weights is rejected.

This is a hypothesis about feature refinement, not a demonstrated root cause of all missed detections. Channel/spatial refinement is established in [CBAM](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.php), and lightweight depthwise operations are described in [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html). Those results do not establish that this SurveilX variant improves accuracy. Weapons and PPE each receive a matching unchanged-architecture control.

## Campaign and safeguards

### Current status: G10d prepared, waiting for commit headroom

The G10c guard stopped its first CUDA batch because Windows commit headroom fell below 512 MiB. No epoch or checkpoint completed. Its runner was stopped before further jobs could train. The controller now halts the entire queue on a child resource failure, even if memory recovers when that child exits.

Dataset integrity validation now lives in `training/detection_manifest.py` and retains its old public imports. This avoids loading PyTorch in the controller. Integrity, label, asset-path, hash and split-leakage checks are unchanged. A subprocess regression forbids importing Torch while validating a fixture. With `OPENBLAS_NUM_THREADS=3`, the controller measured about 112 MiB of memory commitment versus about 648 MiB before these changes. Use `OMP_NUM_THREADS=3` and `MKL_NUM_THREADS=3` too, before interpreter startup.

A no-optimizer CUDA diagnostic at weapons batch two reserved 386 MiB VRAM and left about 1.33 GiB Windows commit headroom. It does not reproduce all trainer/evaluator allocations and does not establish accuracy. Its system samples are recorded in G10c's `commit-probe.json`.

The fresh frozen plan `data/generations/yolo-g10d-lightcontrol-1024/plan.json` uses physical batches two for weapons/fire, one for PPE and four for person, retaining nominal effective batch eight. Accumulation does not replicate larger-batch BatchNorm statistics. The plan requires 3.5 GiB available Windows commit at launch; its final preflight found about 3.03 GiB and **did not start training**. Disk space is sufficient. No automatic retry remains. Close unused applications to free additional commit headroom before restarting; `CONTINUE.md` records the exact safeguards, hashes and commands. Earlier G10d-lowcommit is an unlaunched superseded plan.

Thirty final safe regression tests passed, including the import boundary, queue stopping, dataset leakage, checkpoint safety and compatibility checks. No new accuracy result, acceptance or activation is claimed.

```powershell
.venv/Scripts/python.exe -u -m scripts.watch_training yolo-g10d-lightcontrol-1024
```

### Historical recovery campaign: G10c

The original G10 weapons run completed nine epochs, reaching **0.577610 native validation AP50** at epoch two versus the matching parent **0.543506**. It then failed during epoch ten because Windows could not allocate a CPU tensor. This is a saved validation improvement, not a completed calibrated/test result or proof that the adapter caused the gain.

G10b reduced physical weapons batch size to four (effective eight with accumulation). Its first training epoch and validation completed, but a full disk interrupted checkpoint saving: `best.pt` was empty. Its checkpoints remain preserved for diagnosis and are excluded from continuation.

The new immutable campaign `data/generations/yolo-g10c-guarded-1024/plan.json` starts from the verified original G10 best checkpoint, with a fresh optimizer and schedule. It preserves all previous directories. Weapons uses batch four and a 24-epoch ceiling; the other jobs retain the original settings below. The weapons control is a lineage baseline, **not** an isolated architecture comparison because the local branch already received additional training. PPE retains its matched architecture pair.

Launch requires 3 GiB disk, 3 GiB Windows commit, 1 GiB RAM and 500 MiB free VRAM. Per-batch checks stop below 512 MiB disk or commit headroom. Checkpoints are written completely into a staging directory, flushed, and then atomically replace their public paths; serialization or disk-full failures preserve earlier complete checkpoints. These checks reduce resource failures but cannot guarantee that every allocation will fit.

Storage recovery preserved datasets, model weights and evidence: identical immutable prepared-image copies share hardlinks, and inactive checkpoints/CUDA DLLs use reversible NTFS compression with unchanged byte hashes. Only a redundant downloaded PyTorch installer matching the installed version was removed. The user also freed disk space. Prepared images must remain immutable; copy them before editing.

```powershell
.venv/Scripts/python.exe -u -m scripts.watch_training yolo-g10c-guarded-1024 yolo-g10-local-1024
```

The original G10/G10b plans below are historical and must not be restarted after source changes. No new candidate is automatically accepted or activated.

`data/generations/yolo-g10-local-1024/plan.json` freezes six serial modified-YOLO jobs:

| Job | Parent | Size / physical batch | Epoch ceiling / patience |
|---|---|---|---|
| Weapons local / control | G8c weapons | 512 / 8 | 32 / 8 each |
| PPE local / control | G8c PPE | 640 / 2 | 32 / 8 each |
| Fire local | G9 fire control | 512 / 4 | 24 / 8 |
| Person retention | G2 person | 320 / 8 | 8 / 4 |

Both PPE recipes use `cls_pw=0.35`: the installed Ultralytics implementation computes inverse-frequency weights from the training loader only and normalizes their mean to one. Weapons/fire/person retain unweighted classification. Architecture pairs otherwise share parent, data, seed, learning rate, geometry and schedule. Fire/person are bounded continuation checks, not paired architecture ablations.

CUDA FP32 is required; AMP remains disabled on the MX130. Effective batch size is eight, CPU threads three and data-loader workers zero. Inference/camera workers stay paused. Launch guards retain at least 512 MiB disk, 1 GiB physical RAM, 2 GiB Windows commit and 500 MiB free VRAM before each new job. There are no new datasets or downloads, and no automatic acceptance/activation. Source hashes, data and parent hashes are checked by the existing runner.

The local branch passed dense-label GPU forward/backward probes without optimizer steps or saved weights:

| Probe | Labels/image | Peak reserved | Median forward + backward |
|---|---:|---:|---:|
| Weapons 512, batch 8 | 4 | 1,358 MiB | 0.760 s |
| PPE 640, batch 2 | 154 | 618 MiB | 0.394 s |

These are capacity measurements on repeated training samples, excluding optimizer state, full loading, epochs and evaluation. They are not accuracy or sustained GPU-utilization claims. See `reports/yolo-g10-capacity.json`.

## Evaluation and progress

Checkpoint selection remains native validation AP50, with per-class precision/recall/F1 recorded. Calibration and test remain separate. New artifacts re-evaluate the preserved parent at the candidate's exact size and preprocessing, with calibration fitted separately on calibration data. The runner verifies this paired comparison contract; geometry changes no longer masquerade as learned gains. Test never selects checkpoints. Old artifacts remain unchanged and retain their historical comparison limitations.

Epoch counts now come from the completed trainer epoch, avoiding the extra final-evaluation CSV row. New manifests record epoch-limit/patience stopping and the best epoch. The dashboard and read-only terminal viewer retain existing native histories and loss/LR/per-class diagnostics.

```powershell
.venv/Scripts/python.exe -u -m scripts.watch_training yolo-g10-local-1024
```

Closing the viewer does not stop training. Future interruption recovery must verify the exact plan, processes, source hashes and checkpoint contract. Architecture changes use new version directories; never overwrite or blindly resume G9/G10 outputs. Higher real-world reliability still requires broader reviewed data and independent camera/site evaluation.
