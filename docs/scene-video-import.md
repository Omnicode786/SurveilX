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

The usual training job API also dispatches event manifests to this pipeline. Validation selects the checkpoint; a separate calibration split fits temperature scaling; the test split measures clip classification. Independent acceptance still requires genuinely independent real data and artifact-bound approval. Declaring source independence in metadata does not prove it; source review remains necessary.

At runtime the scene expert resamples timestamped camera observations to the training cadence and abstains when observations are insufficient or too widely spaced. It can return scene evidence without entity detections or zone occupancy. Scene probabilities are experimental evidence, not automatic conclusions of violence, theft or medical injury. A configured `scene_event` policy can now create a review incident only from a calibrated model after its class threshold, persistence, minimum-observation, gap-reset and cooldown checks pass. The incident names the detected class and model evidence; this policy path does not replace independent task validation.

Current limitations: the entity-box annotation workflow cannot assemble scene datasets; use reviewed clip descriptors. Scene ONNX export is explicitly blocked until parity is validated. Real fighting/fall/theft/collision datasets and trained task candidates remain to be acquired and evaluated. The research paper remains excluded.
