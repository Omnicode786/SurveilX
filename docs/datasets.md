# Detection datasets

## Penn-Fudan development dataset

The acquisition recipe downloads the archive directly from the [University of Pennsylvania dataset page](https://www.cis.upenn.edu/~jshi/ped_html/). The publisher describes 170 images of campus and street scenes with 345 labeled upright pedestrians. The downloaded archive contains 423 instances: its README explains that small and occluded pedestrians were added after the original publication. This is the real-image starter dataset for the custom detector and the modified YOLO experiment. It is too small and narrow to establish broad surveillance accuracy or superiority to another detector.

```powershell
.venv/Scripts/python.exe scripts/acquire_dataset.py
```

The command verifies the archive's pinned SHA-256, validates ZIP paths and expansion size, derives person boxes from instance masks, checks their counts against the original annotations, checks for identical-image leakage, and writes these files:

| Path under `data/datasets/pennfudan/` | Purpose |
| --- | --- |
| `PennFudanPed.zip` | Unmodified source archive |
| `PennFudanPed/` | Original images, masks, and annotations |
| `manifest.json` | Custom detector's normalized boxes and split assignments |
| `provenance.json` | Source, hash, acquisition time, rights status, and limitations |
| `dataset.yaml` | YOLO training, validation, and test configuration |
| `calibration.yaml` | Separate post-training calibration configuration |
| `yolo/images/{split}/`, `yolo/labels/{split}/` | YOLO images and normalized center/size labels |

All generated data stays outside version control. An existing manifest is reused only when its acquisition hash and image hashes match. Use a new directory with `--output` for a different dataset version; do not silently replace previous training inputs.

### Rights and attribution

The archive's `PennFudanPed/readme.txt` retains the authors' and other holders' copyright, subjects copying to their terms, and restricts reposting without permission. It contains no standard open-data license or clear commercial-use grant. This copy is recorded as copyright restricted and is used for local development evaluation. Obtain the required rights before redistribution or commercial model deployment. The source is also used in the [official TorchVision detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial), which does not confer a separate dataset license. Provenance records the rights-notice file and its hash.

The publisher's cited work is *Object Detection Combining Recognition and Segmentation*, Liming Wang, Jianbo Shi, Gang Song, and I-fan Shen, ACCV 2007. The acquisition log records attribution without generating a research paper.

### Split boundaries and limitations

Images are ordered separately within the Penn and Fudan filename sequences and divided into contiguous train, validation, calibration, and test bands. One image at each of the three boundaries in each sequence is excluded, leaving 164 usable images. All eight filename-band groups are disjoint across the four splits; calibration images never appear in the normal YOLO training configuration. Exact duplicate hashes cannot cross splits.

| Split | Images | Person boxes |
| --- | ---: | ---: |
| Train | 97 | 245 |
| Validation | 23 | 59 |
| Calibration | 16 | 45 |
| Test | 28 | 58 |

The pinned archive is 53,723,336 bytes with SHA-256 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`.

These filename bands are **approximate sequence groups**, not verified camera or recording-session identities: that metadata is absent. The six guard images reduce immediate adjacency across boundaries, but visual similarities and shared scenes may remain. Results must be described as development-set evidence, not camera-independent or deployment-ready accuracy. The dataset contains no negative images and does not label violence, weapons, actions, or temporal events.

## Bring your own labeled detection data

Create a versioned dataset directory containing the image files and a UTF-8 `manifest.json`:

```json
{
  "schema_version": 1,
  "task": "detection",
  "classes": ["person", "vehicle"],
  "domain": "your-camera-domain",
  "license": "Your documented permission or applicable dataset license",
  "samples": [
    {
      "image": "images/frame-0001.png",
      "boxes": [[0.10, 0.15, 0.45, 0.90]],
      "labels": [0],
      "split": "train",
      "group": "camera-01-session-2026-09-01"
    }
  ]
}
```

`image` is relative to the manifest directory. `boxes` use normalized `[left, top, right, bottom]` coordinates in `[0, 1]`; `labels` contain matching zero-based indices into `classes`. A reviewed negative image uses empty boxes and labels. Include independent `train`, `validation`, `calibration`, and `test` samples; the example above shows one record only. Keep every recording session and adjacent video frames in one group and one split. Use real camera/session IDs when known and independent cameras for deployment evaluation where possible.

Train uses labeled examples; validation selects the checkpoint; calibration chooses post-training confidence behavior; the test set remains untouched until the final evaluation. Unlabeled images and video can be detection inputs and annotation candidates, but automatically predicted labels are not verified ground truth. Review them before training, preserve an immutable dataset version, and require measured improvements before promoting a new model. More data and calibration cannot guarantee perfect accuracy.

For broader models, add licensed, reviewed examples across the intended camera domains, object sizes, lighting, occlusion, backgrounds, and negative scenes. Dataset acquisition and annotation can use CPU resources; CUDA accelerates supported training kernels. FPGA acceleration requires a supported inference/compiler toolchain and compatible model conversion, not a different annotation format.
