# Scene-level video events

Use the scene importer when supervision describes an entire short clip and does not provide entity boxes. It trains the custom `SVASceneNet` frame encoder, temporal GRU and clip-classification head. It does not claim entity localization, frame-level onset labels, trained interactions or real-event accuracy from a synthetic integration test.

Create a JSON descriptor with domain, rights, taxonomy and reviewed clip intervals. For example:

```json
{
  "source_root": "videos",
  "domain": "office-fall-development",
  "license": "Your documented dataset rights",
  "synthetic": false,
  "classes": ["normal", "fall"],
  "capability_ids": ["fall"],
  "source_independence_verified": false,
  "input_contract": {"frames": 8, "entities": 0, "image_size": 64, "clip_seconds": 2},
  "clips": [
    {"video": "session-001.mp4", "start_seconds": 3, "end_seconds": 5, "label": "fall", "split": "train", "group": "subject-01-session-001"}
  ]
}
```

The example illustrates one record; a valid training dataset needs at least ten clips in each of `train`, `validation`, `calibration` and `test`. Whole source videos and source groups must stay in one split. Imported files retain SHA256 bindings to the source videos and resulting clips. The importer samples distinct frames across the exact declared interval and rejects insufficient frame rates or out-of-range intervals. Normal/negative clips must be genuinely labeled rather than inferred from an absence of annotations.

```powershell
.venv/Scripts/python.exe -m training.import_scene descriptor.json data/datasets/office-fall-v1
.venv/Scripts/python.exe -m training.pipeline train data/datasets/office-fall-v1/manifest.json data/runs/office-fall-v1 --epochs 10
.venv/Scripts/python.exe -m scripts.register_run office-fall-v1
```

For ordinary user-provided datasets, the directory helper can create the descriptor and import bounded clips from split/class video folders:

```text
source-root/
  train/normal/*.mp4
  train/collision/*.mp4
  validation/normal/*.mp4
  validation/collision/*.mp4
  calibration/normal/*.mp4
  calibration/collision/*.mp4
  test/normal/*.mp4
  test/collision/*.mp4
```

```powershell
.venv/Scripts/python.exe -m scripts.prepare_scene_event_directory source-root data/datasets/collision-development-v1 --classes normal collision --domain traffic-collision-development --license "documented dataset rights" --capability-id traffic_collision --clip-seconds 2 --frames 8 --image-size 96 --max-clips-per-class-split 50
```

The helper preserves the same validation rules: at least ten samples per split, source videos and groups bound to one split, explicit rights text and explicit capability IDs. Center crops inherit video labels and are marked `inherited_video_label`; this does not establish temporal labels for the interval. Use an explicitly reviewed descriptor for that. Reusing an output requires identical preparation settings; use a new dataset version for changed settings. Dense intervals decode sequentially through native OpenCV, while sparse intervals retain seeking.

The usual training job API also dispatches event manifests to this pipeline. Validation selects the checkpoint; a separate calibration split fits temperature scaling; the test split measures clip classification. Independent acceptance still requires genuinely independent real data and artifact-bound approval. Declaring source independence in metadata does not prove it; source review remains necessary.

At runtime the scene expert resamples timestamped camera observations to the training cadence and abstains when observations are insufficient or too widely spaced. It can return scene evidence without entity detections or zone occupancy. Scene probabilities are experimental evidence, not automatic conclusions of violence, theft or medical injury. A configured `scene_event` policy can now create a review incident only from a calibrated model after its class threshold, persistence, minimum-observation, gap-reset and cooldown checks pass. The incident names the detected class and model evidence; this policy path does not replace independent task validation.

AIRTLab fighting preparation uses `scripts.prepare_airtlab`, a pinned publisher revision and verified Git blob IDs. It selects explicit fight-tag positives and nonviolent hard negatives, groups both camera views of an action into one split and records research/education terms. The 160-source-video subset is staged in one room; center crops inherit video labels without temporal review. Its source is the [official AIRTLab repository](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos). Download/import progress is recorded under `data/downloads/airtlab`. No field accuracy or trained capability follows from this import.

New event artifacts carry source provenance and saved-prediction hashes. `scripts.report_event_uncertainty VERSION DATASET` checks existing prediction labels, frozen dataset and weight integrity, then writes a separate report. Recorded-group resampling preserves paired-view dependence. The Wilson interval concerns the probability that every clip in a recorded group is correct; it is not a clip-accuracy interval when groups contain multiple clips. Neither interval proves independence across sites or actors.

Current limitations: the entity-box annotation workflow cannot assemble scene datasets; use reviewed clip descriptors or the split/class directory helper. Theft/collision candidates and independent event evaluation still require suitable footage and reviewed labels. Acceleration deployment/coverage and complete production verification/infrastructure are excluded from current work. The research paper remains excluded.
