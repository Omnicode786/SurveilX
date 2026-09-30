"""Build a bounded multi-weapon development dataset from Zenodo record 13786228."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

from training.detection_pipeline import digest, validate_detection_manifest

CLASSES = ["firearm", "knife", "machete", "baseball_bat"]
SOURCE_LABELS = {
    "gun": "firearm",
    "rifle": "firearm",
    "knife": "knife",
    "machete": "machete",
    "baseball bat": "baseball_bat",
    "baseball_bat": "baseball_bat",
}
SPLITS = (("train", 0.60), ("validation", 0.15), ("calibration", 0.10), ("test", 0.15))


def parse_annotation(content):
    if len(content) > 1_000_000:
        raise ValueError("Unexpected VOC annotation size")
    upper = content.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("XML entities and DOCTYPE are unsupported")
    record = ET.fromstring(content)
    filename = record.findtext("filename")
    width, height = int(record.findtext("size/width", "0")), int(record.findtext("size/height", "0"))
    if not filename or width <= 0 or height <= 0:
        raise ValueError("VOC filename and positive dimensions are required")
    boxes, labels = [], []
    for obj in record.findall("object"):
        source_label = obj.findtext("name", "").strip().lower()
        if source_label not in SOURCE_LABELS:
            raise ValueError(f"Unknown dangerous-item class: {source_label}")
        if obj.findtext("difficult", "0") == "1":
            raise ValueError("Difficult objects require ignore-region support")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError("VOC object is missing its bounding box")
        values = [float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        normalized = [(values[0] - 1) / width, (values[1] - 1) / height, values[2] / width, values[3] / height]
        if (
            not np.isfinite(normalized).all()
            or values[2] <= values[0]
            or values[3] <= values[1]
            or min(normalized) < -1e-6
            or max(normalized) > 1 + 1e-6
        ):
            raise ValueError("Invalid VOC bounding box")
        boxes.append([min(1.0, max(0.0, value)) for value in normalized])
        labels.append(CLASSES.index(SOURCE_LABELS[source_label]))
    if not labels:
        raise ValueError("Dangerous Items annotations must contain an object")
    return filename, width, height, boxes, labels


def split_bucket(members):
    counts = [int(len(members) * ratio) for _, ratio in SPLITS[:-1]]
    counts.append(len(members) - sum(counts))
    result, offset = [], 0
    for (split, _), count in zip(SPLITS, counts, strict=True):
        result.extend((member, split) for member in members[offset : offset + count])
        offset += count
    return result


def prepare(archive, output, limit_per_class=160):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError("Use a new dataset version directory; existing data is preserved")
    status_path = archive.parent / "acquisition.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "verified_download" or archive.stat().st_size != status.get("bytes"):
        raise ValueError("Verified acquisition record is required")
    if digest(archive) != status.get("sha256"):
        raise ValueError("Archive checksum changed")

    parsed, rejected, source_counts = {}, [], Counter()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not info.filename.startswith("Dangerous Items/annotations/") or not info.filename.endswith(".xml"):
                continue
            try:
                filename, width, height, boxes, labels = parse_annotation(bundle.read(info))
                stem = PurePosixPath(filename).stem
                if stem in parsed:
                    raise ValueError("Duplicate annotation stem")
                image = f"Dangerous Items/images/{filename}"
                image_info = bundle.getinfo(image)
                if image_info.file_size > 20 * 1024**2:
                    raise ValueError("Unexpected image size")
                parsed[stem] = {
                    "source_member": image,
                    "annotation_member": info.filename,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
                source_counts.update(CLASSES[label] for label in labels)
            except (ET.ParseError, KeyError, ValueError) as exc:
                rejected.append({"annotation": info.filename, "reason": str(exc)})

        buckets = defaultdict(list)
        for stem, item in parsed.items():
            signature = tuple(sorted(set(item["labels"])))
            buckets[signature].append((stem, item))
        selected = []
        for signature, members in sorted(buckets.items()):
            members.sort(key=lambda value: hashlib.sha256(("dangerous-items-v1:" + value[0]).encode()).hexdigest())
            cap = limit_per_class * max(1, len(signature))
            selected.extend(split_bucket(members[:cap]))

        output.mkdir(parents=True)
        (output / "images").mkdir()
        samples, hashes = [], set()
        for (stem, item), split in selected:
            content = bundle.read(item["source_member"])
            checksum = hashlib.sha256(content).hexdigest()
            if checksum in hashes:
                rejected.append({"annotation": item["annotation_member"], "reason": "Duplicate selected image content"})
                continue
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if image is None or (image.shape[1], image.shape[0]) != (item["width"], item["height"]):
                rejected.append({"annotation": item["annotation_member"], "reason": "Decoded dimensions disagree with VOC"})
                continue
            hashes.add(checksum)
            suffix = PurePosixPath(item["source_member"]).suffix.lower()
            relative = f"images/{checksum}{suffix}"
            (output / relative).write_bytes(content)
            samples.append(
                {
                    "image": relative,
                    "sha256": checksum,
                    "split": split,
                    "group": f"dangerous-items-unverified-{stem}",
                    "boxes": item["boxes"],
                    "labels": item["labels"],
                    "source_member": item["source_member"],
                    "annotation_member": item["annotation_member"],
                }
            )

    manifest = {
        "schema_version": 1,
        "name": "Dangerous Items bounded development generation 1",
        "task": "detection",
        "domain": "dangerous-items-development",
        "synthetic": False,
        "classes": CLASSES,
        "capability_ids": ["firearm"],
        "license": status["license"],
        "samples": samples,
        "provenance": {
            "source": status["page"],
            "archive_sha256": status["sha256"],
            "source_independence_verified": False,
            "selection": f"Deterministic VOC subset capped at {limit_per_class} images per label signature",
            "group_limitation": "Publisher camera, sequence, subject and original-image identities are unavailable; split groups are unique-image placeholders only",
            "label_mapping": "Publisher Gun and Rifle map to firearm; Knife, Machete and Baseball Bat remain distinct",
            "acceptance_limitation": "Development metrics only; independent site acceptance is required",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_detection_manifest(manifest_path)
    report = {
        "archive_annotations": len(parsed) + len(rejected),
        "valid_annotations": len(parsed),
        "source_boxes": dict(source_counts),
        "selected_images": len(samples),
        "split_counts": dict(Counter(sample["split"] for sample in samples)),
        "selected_boxes": dict(Counter(CLASSES[label] for sample in samples for label in sample["labels"])),
        "rejected": rejected,
        "source_independence_verified": False,
        "production_acceptance_eligible": False,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("output")
    parser.add_argument("--limit-per-class", type=int, default=160)
    args = parser.parse_args()
    if not 10 <= args.limit_per_class <= 2000:
        parser.error("limit-per-class must be 10..2000")
    print(json.dumps(prepare(**vars(args)), indent=2))
