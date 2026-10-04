"""Acquire class-balanced SH17 additions with stable photographer-group splits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import cv2
import numpy as np

from scripts.prepare_sh17 import CLASSES, URL, labels
from scripts.remote_zip import RemoteZip
from scripts.train_domain_generation import save
from training.detection_pipeline import digest, validate_detection_manifest

SPLITS = ("train", "validation", "calibration", "test")


def split_for(photographer):
    group = f"sh17-photographer-{photographer}"
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return group, "train" if bucket < 60 else "validation" if bucket < 75 else "calibration" if bucket < 85 else "test"


def index_source(output):
    output = Path(output)
    if output.exists():
        return json.loads(output.read_text())
    remote = RemoteZip(URL, budget=64 * 1024**2)
    records, rejected = [], []
    with zipfile.ZipFile(remote) as bundle:
        entries = bundle.infolist()
        annotations = [item for item in entries if item.filename.startswith(("labels/", "meta-data/"))]
        first = min(item.header_offset for item in annotations)
        last = max(item.header_offset + item.compress_size + 4096 for item in annotations)
        # The publisher stores small annotations contiguously. One bounded request avoids
        # thousands of requests while ZipFile still verifies every member's CRC.
        remote.cache_start, remote.cache = first, remote._range(first, last)
        images = {PurePosixPath(item.filename).stem: item for item in entries
                  if item.filename.startswith("images/") and item.filename.lower().endswith((".jpg", ".jpeg", ".png"))}
        for stem, image in sorted(images.items()):
            try:
                raw_label = bundle.read(f"labels/{stem}.txt")
                raw_meta = bundle.read(f"meta-data/{stem}.json")
                boxes, categories = labels(raw_label.decode("utf-8"))
                metadata = json.loads(raw_meta)
                photographer = metadata.get("photographer_id")
                if not isinstance(photographer, int) or photographer <= 0:
                    raise ValueError("Missing publisher photographer identity")
                if not categories or image.file_size > 16 * 1024**2:
                    raise ValueError("Empty labels or source image exceeds 16 MiB bound")
                group, split = split_for(photographer)
                records.append({"source_member": image.filename, "boxes": boxes, "labels": categories,
                                "group": group, "split": split, "source_url": metadata.get("url"),
                                "source_bytes": image.file_size,
                                "label_sha256": hashlib.sha256(raw_label).hexdigest(),
                                "metadata_sha256": hashlib.sha256(raw_meta).hexdigest()})
            except (ValueError, KeyError) as exc:
                rejected.append({"image": image.filename, "reason": str(exc)})
    result = {"url": URL, "etag": remote.etag, "archive_bytes": remote.size, "records": records,
              "rejected": rejected, "transferred_bytes": remote.transferred,
              "availability": {split: dict(Counter(CLASSES[label] for row in records if row["split"] == split
                                                    for label in set(row["labels"]))) for split in SPLITS}}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    return result


def select_additions(records, existing, targets, maximum):
    """Greedy coverage of rare classes; only labels/group metadata drive selection."""
    done = {row["source_member"] for row in existing}
    counts = {split: Counter(label for row in existing if row["split"] == split for label in set(row["labels"]))
              for split in SPLITS}
    available = [row for row in records if row["source_member"] not in done]
    available.sort(key=lambda row: hashlib.sha256(row["source_member"].encode()).hexdigest())
    membership = np.zeros((len(available), len(CLASSES)), dtype=np.float32)
    split_names = list(targets)
    split_ids = np.array([split_names.index(row["split"]) for row in available], dtype=int)
    for index, row in enumerate(available):
        membership[index, list(set(row["labels"]))] = 1
    available_mask = np.ones(len(available), dtype=bool)
    selected = []
    while available_mask.any() and len(selected) < maximum:
        # Rare labels receive a larger weight while plentiful person examples do not dominate.
        weights = np.array([[max(targets[split] - counts[split][label], 0) / (1 + counts[split][label])
                             for label in range(len(CLASSES))] for split in split_names])
        scores = (membership * weights[split_ids]).sum(axis=1)
        scores[~available_mask] = -1
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        item = available[best]
        available_mask[best] = False
        selected.append(item)
        counts[item["split"]].update(set(item["labels"]))
    return selected


def prepare(output, index_path, maximum=1200):
    output, index_path = Path(output), Path(index_path)
    base_path = Path("data/datasets/sh17-development-v1/manifest.json")
    base, _, _ = validate_detection_manifest(base_path)
    source = index_source(index_path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        validate_detection_manifest(output / "manifest.json")
        return {"state": "already_complete"}
    config = {"base_sha256": digest(base_path), "index_sha256": digest(index_path), "maximum_additions": maximum,
              "target_images_per_class": {"train": 100, "validation": 25, "calibration": 20, "test": 25}}
    status_path = output / "acquisition.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "config": config, "samples": [], "rejected": [], "source": {key: source[key] for key in ("url", "etag", "archive_bytes")}}
    if state["config"] != config:
        raise ValueError("Acquisition configuration changed; preserve this output and use another version")
    (output / "images").mkdir(exist_ok=True)
    if not state["samples"]:
        for sample in base["samples"]:
            shutil.copy2(base_path.parent / sample["image"], output / sample["image"])
        state["samples"] = base["samples"].copy()
        save(status_path, state)
    for sample in state["samples"]:
        if digest(output / sample["image"]) != sample["sha256"]:
            raise ValueError("Previously acquired image checksum changed")
    selected = select_additions(source["records"], base["samples"], config["target_images_per_class"], maximum)
    save(output / "selection.json", {"config": config, "members": [item["source_member"] for item in selected]})
    done = {row["source_member"] for row in state["samples"]} | {row["image"] for row in state["rejected"]}
    source_hashes = {row.get("source_sha256", row["sha256"]) for row in state["samples"]}
    remote = RemoteZip(URL, budget=4 * 1024**3)
    if remote.etag != source["etag"]:
        raise ValueError("Source archive ETag changed after annotation indexing")
    try:
        with zipfile.ZipFile(remote) as bundle:
            for row in selected:
                member = row["source_member"]
                if member in done:
                    continue
                if shutil.disk_usage(output).free < 4 * 1024**3:
                    raise ValueError("Preserving 4 GiB free disk reserve")
                content = bundle.read(member)
                source_hash = hashlib.sha256(content).hexdigest()
                pixels = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if source_hash in source_hashes or pixels is None:
                    state["rejected"].append({"image": member, "reason": "Duplicate source content or invalid image"})
                    save(status_path, state)
                    continue
                # Preserve aspect ratio and normalized boxes. Retain source hashes and CRC provenance.
                scale = min(1.0, 1536 / max(pixels.shape[:2]))
                if scale < 1:
                    pixels = cv2.resize(pixels, (round(pixels.shape[1] * scale), round(pixels.shape[0] * scale)),
                                        interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("Could not encode acquired image")
                checksum = hashlib.sha256(encoded.tobytes()).hexdigest()
                relative = f"images/{checksum}.jpg"
                (output / relative).write_bytes(encoded.tobytes())
                state["samples"].append({**row, "image": relative, "sha256": checksum, "source_sha256": source_hash})
                source_hashes.add(source_hash)
                state.update(state="downloading", acquired_additions=len(state["samples"]) - len(base["samples"]),
                             planned_additions=len(selected), transferred_this_run=remote.transferred)
                save(status_path, state)
                print(json.dumps({key: state[key] for key in ("state", "acquired_additions", "planned_additions")}), flush=True)
        manifest = {**base, "name": "SH17 balanced development generation 2", "samples": state["samples"],
                    "provenance": {**base["provenance"], "parent_dataset_sha256": config["base_sha256"],
                                   "selection": "Class-balanced label coverage; stable photographer group assignment; all previous samples keep their split",
                                   "source_index_sha256": config["index_sha256"],
                                   "image_processing": "New images preserve aspect ratio with max side 1536 and JPEG quality 95; original source hashes retained",
                                   "evaluation_change": "Expanded development holdouts; compare parent and child on this identical dataset, not old test scores"}}
        save(output / "manifest.json", manifest)
        _, counts, _ = validate_detection_manifest(output / "manifest.json")
        state.update(state="completed", counts=counts)
        state.pop("error", None)
        save(status_path, state)
        return {"state": state["state"], "counts": counts}
    except Exception as exc:
        state.update(state="interrupted", error=str(exc))
        save(status_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--index", default="data/downloads/sh17/label-index.json")
    parser.add_argument("--maximum", type=int, default=1200)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.maximum <= 4000:
        parser.error("maximum must be 1..4000")
    if args.index_only:
        indexed = index_source(args.index)
        print(json.dumps({key: indexed[key] for key in ("availability", "transferred_bytes")}, indent=2))
    else:
        print(json.dumps(prepare(args.output, args.index, args.maximum), indent=2))
