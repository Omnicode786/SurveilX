# Dataset compatibility

The supported tasks are object detection and video-event classification. Dataset format support does not mean pretrained recognition of every class: each model version learns the taxonomy supplied in its manifest.

| Input | Supported supervision | Route | Boundary |
|---|---|---|---|
| SurveilX detection manifest + images | Any explicit class taxonomy, axis-aligned boxes, negative images, optional scene/zone inputs | `training.detection_pipeline` or `training.yolo_pipeline` | Four nonempty independent splits required |
| COCO JSON + images | Bounding boxes and category IDs mapped by class name | `training.import_detection`, format `coco` | Crowd/ignore semantics rejected; masks/keypoints not trained |
| YOLO image/label directories | Five-column class/cx/cy/w/h detection rows | Same converter, format `yolo` | Explicit empty TXT for verified negative images; polygons/pose rejected |
| Pascal VOC XML + images | Object names and one-based inclusive boxes | Same converter, format `voc` | Difficult/ignore objects rejected; curate full images explicitly |
| Annotated video files | Per-frame persistent entity boxes and single clip-level event label | `training.import_data` | Fixed contract within a version, no automatic ground-truth labeling |
| NPZ event clips | RGB clip, persistent boxes, four scene values and event label | `training.pipeline` | Default 8 frames/2 entities/64px; explicit contract can change these |
| Scene video/NPZ clips | Whole-clip event label without invented entity boxes | `training.import_scene` and `training.pipeline` | Fixed duration/cadence and whole-video group/hash separation; runtime abstains when cadence is invalid |
| Dangerous Items VOC archive | Gun/Rifle mapped to `firearm`; Knife, Machete and Baseball Bat preserved | `scripts.prepare_dangerous_items` | Development-only split because publisher source identities are unavailable |
| Unlabeled images/video | Inference and subsequent human review | Cameras/media + feedback | Model guesses do not become verified labels automatically |
| Segmentation, pose, audio, text-only labels | Not supported in this scope | Additional task adapter/head required | Explicitly deferred by user preference |

## Common-format import

Save a descriptor alongside your source directories. Each source needs its own explicit split and source group. Use actual camera/session identifiers, not arbitrary per-image IDs. If an export contains several sessions, add `groups: {"image-name.png": "session-id"}` to its source.

```json
{
  "format": "yolo",
  "classes": ["person", "vehicle"],
  "domain": "warehouse-day",
  "license": "Organization-owned; authorized training use",
  "sources": [
    {"split":"train", "group":"session-a", "images":"train/images", "annotations":"train/labels"},
    {"split":"validation", "group":"session-b", "images":"validation/images", "annotations":"validation/labels"},
    {"split":"calibration", "group":"session-c", "images":"calibration/images", "annotations":"calibration/labels"},
    {"split":"test", "group":"session-d", "images":"test/images", "annotations":"test/labels"}
  ]
}
```

For `coco`, each `annotations` value is a COCO JSON file. For `voc`, it is an XML directory. All referenced paths must remain below the descriptor directory.

```powershell
.\.venv\Scripts\python -m training.import_detection incoming/import.json data/datasets/site-v1
```

Conversion validates geometry, image dimensions, source groups, taxonomy and content hashes before creating the destination. The canonical dataset can also be zipped with `manifest.json` at the root and imported in the command center. ZIP imports accept JSON, NPZ, PNG and JPEG and enforce traversal/file-size limits.

Select **Start training → architecture** to train the scratch model, the modified YOLO model, or automatically choose the event/detection pipeline. Completion includes validation checkpoint selection, held-out calibration, test evaluation and candidate registration. Models remain candidates until explicitly activated as a canary; production needs independent acceptance.

## Video-event contract

The video annotation format is in [data and training](data-and-training.md). Add an optional top-level contract:

```json
{"input_contract": {"frames": 16, "entities": 4, "image_size": 96}}
```

Provide 16 frame indices, a `[16,4,4]` box array in persistent entity order, one of the manifest's event classes and four context values per sample. Training permits 2–64 frames, 1–32 entities and square images 32–512 pixels per side. Larger contracts increase memory and pairwise-relation cost. All clips in a version must match its contract. Each split needs at least ten samples; calibration requires at least two represented event classes. Untracked/missing entities require curation, because padding masks are not supported. For scene-level events without entity tracks, use `representation: scene_clip`, `entities: 0` and the scene importer described in [scene-video import](scene-video-import.md).

Real datasets such as UCF-Crime do not automatically satisfy this entity-centric contract just because they contain videos. Their event labels, temporal boundaries, rights, source grouping and persistent boxes must be adapted and reviewed. Do not mix class definitions or treat missing annotations as negative examples.

## Adding another dataset

First identify its task, rights, label semantics, coordinate convention and grouping information. Implement an adapter to the existing canonical contract where those semantics match. Otherwise add a versioned task/head and evaluator. Reuse the common validators and four-split pipeline; do not bypass them to make a dataset appear supported.

Format references: [COCO API](https://github.com/cocodataset/cocoapi), [Ultralytics detection format](https://docs.ultralytics.com/datasets/detect/), [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
