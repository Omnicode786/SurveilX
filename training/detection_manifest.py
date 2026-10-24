"""Detection dataset integrity checks without importing model/training runtimes."""
import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "calibration", "test")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolve_asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Dataset asset is missing or outside the dataset root: {relative}")
    return path


def validate_detection_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    from surveilx.task_profiles import validate_profiles

    validate_profiles(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("task") != "detection":
        raise ValueError("Expected a schema_version=1 detection manifest")
    if not manifest.get("domain") or not manifest.get("license"):
        raise ValueError("Detection datasets require a domain and license/rights record")
    classes = manifest.get("classes", [])
    if not classes or len(classes) != len(set(classes)) or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("classes must contain distinct nonempty class names")
    if not manifest.get("samples"):
        raise ValueError("The dataset has no labeled samples")
    counts, groups, images, hashes = dict.fromkeys(SPLITS, 0), {}, {}, {}
    for sample in manifest["samples"]:
        split, group = sample.get("split"), sample.get("group")
        if split not in SPLITS or not isinstance(group, str) or not group:
            raise ValueError("Every sample needs an explicit split and source group")
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group leaks across dataset splits: {group}")
        groups[group] = split
        image = resolve_asset(path.parent, sample["image"])
        image_hash = digest(image)
        if image_hash in images and images[image_hash] != split:
            raise ValueError("Identical image content leaks across dataset splits")
        if sample.get("sha256") and sample["sha256"] != image_hash:
            raise ValueError(f"Image checksum mismatch: {sample['image']}")
        images[image_hash], hashes[sample["image"]] = split, image_hash
        boxes = np.asarray(sample.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(sample.get("labels", []))
        if len(boxes) != len(labels) or not np.isfinite(boxes).all():
            raise ValueError("Each box needs a class label and finite coordinates")
        if len(boxes) and (np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Bounding boxes must be normalized xyxy with positive area")
        if len(labels) and (
            not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= len(classes))
        ):
            raise ValueError("Class labels must be zero-based integers from classes")
        context = np.asarray(sample.get("context", [0, 0, 0, 0]), dtype=np.float32)
        if context.shape != (4,) or not np.isfinite(context).all():
            raise ValueError("context must have exactly four finite numbers")
        if sample.get("zone_map"):
            zone = resolve_asset(path.parent, sample["zone_map"])
            hashes[sample["zone_map"]] = digest(zone)
        counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Training requires nonempty train, validation, calibration, and test splits")
    return manifest, counts, hashes


