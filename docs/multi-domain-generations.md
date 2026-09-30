# Multi-domain generations

The task coverage page (`/capabilities`) reports each requested detection, video-event and site-policy family separately. Imported datasets and trained candidates are shown with their domain, calibration, approval and live activation. A declared class or profile is not evidence of real-world accuracy. Production approval is checked against the actual artifact and acceptance record.

## Fire/smoke development generation

The downloaded D-Fire archive contains 21,527 images. The importer validates all source label files, excludes invalid boxes, retains negative images and creates a bounded 672-image development dataset. See `data/datasets/dfire-development-v1/audit.json` for rejected members and counts.

The four roles contain 384 training images and 96 images each for validation, calibration and testing. The importer separates filename bands within each source family, omits 32 images at each band boundary, samples label combinations deterministically, checks selected image hashes and verifies decoding. This does **not** establish camera/session independence: those identities are unavailable. Development scores must not be presented as independent site acceptance. The acceptance evaluator explicitly rejects this dataset as independent acceptance evidence.

The source archive and attribution are recorded in `data/downloads/dfire/acquisition.json`. The author repository describes collection licensing and underlying-image rights limitations: [D-Fire source](https://github.com/gaia-solutions-on-demand/DFireDataset).

## Running and resuming generations

From the repository root:

```powershell
.venv/Scripts/python.exe -m scripts.train_domain_generation dfire-development-v1 dfire-g1 --epochs 3 --threads 3
```

This runs scratch SVA-Detector followed by adapted YOLO serially. Each pipeline trains, uses validation for checkpoint selection, fits calibration on its separate split and evaluates the held-out development test split. Completed artifacts are registered as candidates; the runner never activates them.

`data/generations/<generation>/status.json` records commands, log paths, states and model IDs. A process lock prevents duplicate runners. Completed artifacts are reused after integrity verification. Partial runs are preserved and produce an explicit failure instead of being overwritten. After a machine crash, check the recorded PID and command line before removing a stale lock. An adapted-YOLO run may resume only from its own verified `fit/weights/last.pt` with an unchanged dataset and training configuration. Event and scratch adaptation jobs can warm-start only from a registry-bound parent whose task, taxonomy, input contract, architecture and weight hash all match. Each completed adaptation writes a retention report against the unchanged non-train samples; independent acceptance is still required.

## PPE source audit

[SH17's author repository](https://github.com/ahmadmughees/SH17dataset) provides a 17-class PPE/body-part dataset and links its Kaggle archive. The source declares CC BY-NC-SA 4.0 and additional underlying Pexels conditions. Its `Glasses` label must not automatically be relabeled as certified safety eyewear.

A bounded 256-image subset was acquired using verified HTTP ranges and ZIP CRC checks, transferring 452,651,139 bytes. The 14 GB archive was not downloaded in full. Photographer IDs stay within one of the four splits (163/29/25/39). Site independence remains unverified. Numeric labels follow the author's `sh17.yaml`, not the README display order; the aborted g1 run is preserved and corrected g2 is the usable generation. The scratch candidate reached AP50 0.0135705. The crash-resumed adapted candidate reached AP50 0.0476803, precision 0.566265 and recall 0.129477; only its person class was useful, while PPE-class AP50 values were zero on this small test split. Both candidates remain inactive and inadequate for PPE compliance.

## Fall-event development generation

The [UR Fall publisher dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) supplies original RGB sequences under CC BY-NC-SA 4.0 for noncommercial academic use. `scripts/prepare_urfall.py` performs range-based ZIP extraction with CRC and source-member hashes, keeps each sequence in one split and creates 40 fixed-duration scene clips: 10 each for training, validation, calibration and testing, balanced between normal activity and falls. Participant and site identities are unavailable, so the manifest explicitly denies independent-source verification.

`urfall-g1` trained and calibrated the custom scene-event architecture and registered model `094c00ac-b631-4a9b-b10d-36f5f6cedd93`. Held-out accuracy was 0.50 on 10 clips, so it is an inactive development candidate and does not establish reliable fall detection. The event-policy path can persist calibrated scene classes and record the detected label, confidence, model and evidence for operator review.

## Firearm data acquisition

The [Dangerous Items dataset](https://zenodo.org/records/13786228) is a CC BY 4.0 VOC-style source containing gun, rifle, knife, machete and baseball-bat annotations. Its 357,834,955-byte archive is being downloaded resumably. The tested bounded importer maps Gun and Rifle to canonical `firearm`, keeps Knife, Machete and Baseball Bat as explicit detected classes, rejects unknown/difficult/invalid annotations and marks source independence as unavailable. No firearm training manifest or candidate may be claimed until the archive is fully verified and the importer has completed its audit.

## Remaining scope

Named specialist slots and scene-level imports/models are implemented. Fall now has a real but inaccurate development candidate. Firearm acquisition is incomplete; fighting, theft and traffic event datasets still lack trained candidates. Full traffic/industrial policy validation, independent acceptance, accelerator execution and longitudinal generation evaluation remain open. The research paper is excluded. `CONTINUE.md` is the authoritative resume checklist.
