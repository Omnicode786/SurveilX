"""Resumable, bounded SH17 development subset; no source-independence claim."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import cv2
import numpy as np

from scripts.remote_zip import RemoteZip
from training.detection_pipeline import digest, validate_detection_manifest

URL = "https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection"
CLASSES = [
    "person",
    "ear",
    "earmuffs",
    "face",
    "face_guard",
    "face_mask",
    "foot",
    "tools",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical_suit",
    "shoes",
    "safety_suit",
    "safety_vest",
]


def labels(text):
    boxes, categories = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or not parts[0].isdigit() or not 0 <= int(parts[0]) < len(CLASSES):
            raise ValueError("Invalid SH17 label taxonomy")
        x, y, w, h = map(float, parts[1:])
        box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        if not np.isfinite(box).all() or w <= 0 or h <= 0 or min(box) < -1e-6 or max(box) > 1 + 1e-6:
            raise ValueError("Invalid normalized SH17 box")
        boxes.append([min(1.0, max(0.0, value)) for value in box])
        categories.append(int(parts[0]))
    return boxes, categories


def prepare():
    root = Path("data/datasets/sh17-development-v1")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        validate_detection_manifest(root / "manifest.json")
        return {"state": "already_complete"}
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise ValueError("Need 4 GiB free to preserve the storage reserve")
    status_path = root / "acquisition.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"samples": [], "rejected": []}
    remote = RemoteZip(URL)
    if state.get("etag") and state["etag"] != remote.etag:
        raise ValueError("Source archive changed; existing subset preserved")
    state.update(etag=remote.etag, state="downloading", archive_bytes=remote.size, url=URL)

    def save():
        state["transferred_this_run"] = remote.transferred
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    save()
    try:
        with zipfile.ZipFile(remote) as bundle:
            images = sorted(
                i.filename
                for i in bundle.infolist()
                if i.filename.startswith("images/")
                and i.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                and i.file_size <= 3 * 1024**2
            )
            images.sort(key=lambda name: hashlib.sha256(("sh17-v1:" + name).encode()).hexdigest())
            selected = images[:256]
            done = {sample["source_member"] for sample in state["samples"]}
            rejected = {item["image"] for item in state["rejected"]}
            hashes = {sample["sha256"] for sample in state["samples"]}
            (root / "images").mkdir(exist_ok=True)
            for sample in state["samples"]:
                if digest(root / sample["image"]) != sample["sha256"]:
                    raise ValueError("Previously downloaded subset image changed")
            for name in selected:
                if name in done or name in rejected:
                    continue
                metadata_name = "meta-data/" + PurePosixPath(name).stem + ".json"
                if bundle.getinfo(metadata_name).file_size > 100000:
                    raise ValueError("Unexpected source metadata size")
                metadata = json.loads(bundle.read(metadata_name))
                photographer = metadata.get("photographer_id")
                if not isinstance(photographer, int) or photographer <= 0:
                    raise ValueError("Verified publisher photographer ID required for grouped splitting")
                group = f"sh17-photographer-{photographer}"
                bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
                split = (
                    "train"
                    if bucket < 60
                    else "validation"
                    if bucket < 75
                    else "calibration"
                    if bucket < 85
                    else "test"
                )
                label_name = "labels/" + PurePosixPath(name).stem + ".txt"
                if bundle.getinfo(label_name).file_size > 100000:
                    raise ValueError("Unexpected label size")
                try:
                    boxes, categories = labels(bundle.read(label_name).decode("utf-8"))
                except ValueError as exc:
                    state["rejected"].append({"image": name, "reason": str(exc)})
                    save()
                    continue
                content = bundle.read(name)
                checksum = hashlib.sha256(content).hexdigest()
                if checksum in hashes:
                    state["rejected"].append({"image": name, "reason": "Identical selected image"})
                    save()
                    continue
                if cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError(f"Cannot decode source image: {name}")
                relative = f"images/{checksum}{PurePosixPath(name).suffix}"
                (root / relative).write_bytes(content)
                hashes.add(checksum)
                state["samples"].append(
                    {
                        "image": relative,
                        "sha256": checksum,
                        "split": split,
                        "group": group,
                        "source_url": metadata.get("url"),
                        "boxes": boxes,
                        "labels": categories,
                        "source_member": name,
                    }
                )
                save()
                print(f"SH17: {len(state['samples'])}/{len(selected)} images", flush=True)
        manifest = {
            "schema_version": 1,
            "name": "SH17 bounded development generation 1",
            "task": "detection",
            "domain": "ppe-development",
            "synthetic": False,
            "classes": CLASSES,
            "capability_ids": ["ppe_objects"],
            "license": "SH17 CC BY-NC-SA 4.0; underlying Pexels conditions; noncommercial development only",
            "samples": state["samples"],
            "provenance": {
                "source": "https://github.com/ahmadmughees/SH17dataset",
                "archive_etag": remote.etag,
                "label_map_source": "https://github.com/ahmadmughees/SH17dataset/blob/master/sh17.yaml",
                "source_independence_verified": False,
                "selection": "Seeded subset of images <=3 MiB; publisher photographer IDs assigned wholly to one split; selection bias not evaluated",
                "group_limitation": "Photographer-disjoint development split, not verified scene or deployment-site independence",
                "label_semantics": "Glasses is generic eyewear, not certified safety eyewear. Object presence does not establish compliance.",
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        validate_detection_manifest(manifest_path)
        state.update(state="completed", split_counts=dict(Counter(s["split"] for s in state["samples"])))
        save()
        return {"state": "completed", "images": len(state["samples"]), "split_counts": state["split_counts"]}
    except Exception as exc:
        state.update(state="interrupted", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
