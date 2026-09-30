# Data, training and calibration

## One pipeline, four independent data splits

1. Generate test samples or import reviewed video annotations.
2. Validate shapes, labels, licenses, checksums and split isolation.
3. Train on `train`; select the best epoch using `validation` loss.
4. Fit temperature only on `calibration` logits.
5. Report accuracy, Brier score, ECE and risk–coverage on untouched `test` samples.
6. Save weights, predictions, configuration, software versions and a run manifest.
7. Register a candidate. Synthetic results never establish production fitness.

The command center can generate a dataset and start the entire train/evaluate/calibrate/register sequence. Run the same workflow from the command line:

```powershell
.\.venv\Scripts\python -m training.pipeline generate data/datasets/generated-v2 --count 400 --seed 71
.\.venv\Scripts\python -m training.pipeline train data/datasets/generated-v2/manifest.json data/runs/run-v2 --epochs 25
.\.venv\Scripts\python -m training.export data/runs/run-v2
```

Generated samples are small, colored moving shapes with slow/fast-motion labels. They exercise software and trainable gradients, not real surveillance understanding. Four live `demo://0` through `demo://3` sources are also available; their color detector is explicitly a test fixture.

## Import your own video data

Use `python -m training.import_data annotations.json data/datasets/site-v1`. Keep referenced videos beside or below the annotation file. By default, each sample specifies eight frame indices and two persistent entity boxes per frame. A version can declare another frame/entity/image-size contract; see [dataset adapters](dataset-adapters.md). Missing-entity masks and supervised relation labels remain unsupported.

```json
{
  "domain": "warehouse-day",
  "license": "Organization-owned footage; approved research use",
  "classes": ["routine", "reviewable_interaction"],
  "samples": [
    {
      "video": "clips/camera-a-session-1.mp4",
      "frame_indices": [0, 4, 8, 12, 16, 20, 24, 28],
      "boxes": "Replace with an 8 by 2 by 4 numeric array of normalized xyxy boxes",
      "context": [1, 0, 0, 0],
      "label": 0,
      "group": "camera-a-session-1",
      "split": "train"
    }
  ]
}
```

The abbreviated boxes field above explains the shape and is not executable annotation data. Provide at least ten samples in each split and both classes for calibration. Entire camera/session groups must stay within one split. Keep a genuinely external site/time period for final domain-transfer evaluation. Generated seeds alone do not create real-world domain independence.

Alternatively provide `manifest.json` plus `.npz` files directly under `data/datasets/<version>/`. Each NPZ contains float32 `clip` (8,3,64,64) RGB in [0,1], `boxes` (8,2,4) normalized xyxy, and `context` (4,). Each sample contains `file`, `label`, `group`, `split`, and optional verified `sha256`. The manifest includes `schema_version: 1`, `name`, `synthetic`, `license`, `domain`, `classes`, and `samples`. Use immutable version directories.

## Feedback and later generations

Operators can submit true-event, false-positive, ambiguous and missed-event labels. Another user must review the label. The evidence annotation editor adds object boxes, persistent tracks and video-event labels; a second person reviews these before dataset assembly. **Reviewed learning** builds new versions with train-only approved evidence, bounded replay and unchanged evaluation splits, and can automatically train/calibrate candidates after an approval threshold. See the [complete workflow and limits](reviewed-learning.md).

For event-model adapter-only updates from the CLI, pass `--initialize-from data/runs/previous/weights.pt`; only context and event-head parameters update. CLI continuation uses the samples supplied in its manifest. Dashboard automatic generations currently retrain their selected architecture with explicitly recorded replay rather than automatically continuing previous weights. Evaluate old-domain retention separately. Never add validation/calibration/test samples to replay training.

SVA-Net currently trains its event head with categorical cross entropy; entity-state and relation heads are present but untrained because those labels have not been provided. They must not be interpreted as trustworthy output. The modified YOLO detector is integrated and trained on pedestrian boxes with neutral context; supervised risk conditioning is not validated. See the [real-image benchmark](detection-benchmark.md).

## Deployment acceptance

Perfect accuracy is not a valid acceptance promise. Define minimum recall, maximum false alarms/hour, abstention coverage, critical-event miss tolerance and latency for the intended site before evaluation. Use confidence intervals, adequate rare-event sample counts, independent evaluation and operator review. A low ECE alone is insufficient: an uninformative 50/50 classifier can be well calibrated.

**Model acceptance** runs frozen-model evaluations on an independent dataset and checks training ancestry for group/content overlap. Its current automated criteria cover support, source diversity, AP50/accuracy, Wilson lower bounds, ECE, wall time and optional baseline regression. It does not yet measure false alarms/hour. A passing real-domain report requires explicit approval, scoped canary activation and production promotion. Changes to accepted weights or calibration invalidate artifact approval.

## Dataset acquisition plan

Penn-Fudan has been downloaded and converted for local pedestrian-detection evaluation, with provenance and rights limitations in [dataset notes](datasets.md). COCO-format boxes can be imported; MOT/CityFlow tracking and UCF-Crime event annotations require task-specific adaptation. Do not merge datasets into one label space without a mapping. Custom footage collection and real event annotation have not been performed here. Audit each source's license, permissions, taxonomy, weak versus dense labels and leakage before import.
