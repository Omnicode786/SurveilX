# Initial real-image detection benchmark

Three candidates use the same Penn-Fudan manifest: 97 training, 23 validation, 16 calibration and 28 test images, with 58 person boxes in the test split. Source grouping and rights limitations are in [dataset notes](datasets.md). These data are too small and narrow for general surveillance claims.

| Candidate | Initialization | Epochs | Parameters | Test AP50 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| SVA-Detector | Random; custom width 24, depth 2 | 10 | 1,402,014 | 0.6249 | 0.3805 | 0.7414 | 0.5029 |
| YOLO11n baseline | Inherited pretrained weights | 5 | See run manifest | 0.8493 | 0.7656 | 0.8448 | 0.8033 |
| YOLO11n with scene/zone adapters | Same pretrained source | 5 | 2,620,790 | 0.8645 | 0.8909 | 0.8448 | 0.8673 |

**The scratch model did not beat YOLO.** The adapter variant's small advantage in one run does not establish reliable superiority. Both YOLO candidates benefit from external pretraining; scratch versus pretrained is a practical comparison, not a controlled equal-data architecture study. Initialization, losses, augmentation, optimizer and training duration differ. Context was zero because this dataset has no scene/zone labels.

All models use the common 101-point AP50 evaluator, normalized boxes, 256px square evaluation images, score floor 0.001, NMS IoU 0.5 and maximum 100 predictions. Precision/recall/F1 use thresholds fitted on the separate calibration split. Validation selects checkpoints; test images select neither weights nor thresholds. Detection calibration targets box/class correctness at IoU 0.5 and is not incident/threat calibration.

The compact comparison also recomputes held-out ECE/Brier/NLL over retained predictions. ECE decreased from 0.0928 to 0.0068 for scratch, 0.0116 to 0.0050 for baseline YOLO, and 0.0149 to 0.0062 for adapted YOLO. These prediction-level measures can be dominated by low-score negatives and omit undetected objects, so they must be read alongside recall and false-negative counts. They do not constitute domain-transfer acceptance.

Runs are under `data/runs/scratch-pennfudan-v1`, `yolo-baseline-pennfudan-v1` and `yolo-rai-pennfudan-v1`. They retain predictions, configuration, manifest hashes, weights and metrics. YOLO runs also retain upstream training logs. `reports/detection-comparison.json` records the compact comparison.

CPU training times from overlapping local jobs are not clean speed comparisons. The scratch manifest's batch-amortized latency and YOLO's component timings have different measurement scope; do not compare them as identical end-to-end measurements.

## Reproduce

```powershell
.\.venv\Scripts\python scripts/acquire_dataset.py
.\.venv\Scripts\python -m training.detection_pipeline train data/datasets/pennfudan/manifest.json data/runs/scratch-new --epochs 10 --threads 3
.\.venv\Scripts\python -m training.yolo_pipeline train data/datasets/pennfudan/manifest.json data/runs/baseline-new --epochs 5 --baseline
.\.venv\Scripts\python -m training.yolo_pipeline train data/datasets/pennfudan/manifest.json data/runs/adapted-new --epochs 5
```

Future claims require more independent scenes and classes, multiple seeds, uncertainty intervals, matched preprocessing/compute, real event datasets, calibrated domain transfer and target-hardware latency/power. Preserve this test set as an exploratory benchmark once its results inform future model changes; use a new untouched acceptance set.
