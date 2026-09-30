"""Build a bounded D-Fire development generation, without claiming source independence."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import cv2
import numpy as np

from training.detection_pipeline import validate_detection_manifest


def parse_labels(text):
    boxes, labels = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in ("0", "1"):
            raise ValueError("Expected D-Fire smoke/fire YOLO labels")
        x, y, w, h = map(float, fields[1:])
        box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        if not np.isfinite(box).all() or w <= 0 or h <= 0 or min(box) < -1e-6 or max(box) > 1 + 1e-6:
            raise ValueError("Invalid normalized bounding box")
        boxes.append([min(1.0, max(0.0, value)) for value in box])
        labels.append(int(fields[0]))
    return boxes, labels


def filename_bands(names, guard=32):
    """Conservative filename bands are development proxies, not verified scene identities."""
    groups = defaultdict(list)
    for name in names:
        stem = PurePosixPath(name).stem
        match = re.fullmatch(r"(AoF|WEB|PublicDataset)(\d+)", stem)
        if not match:
            raise ValueError(f"Unknown source naming scheme: {stem}")
        groups[match[1]].append((int(match[2]), name))
    result = {}
    for family, values in sorted(groups.items()):
        ordered = [name for _, name in sorted(values)]
        available = len(ordered) - guard * 3
        if available < 40:
            raise ValueError("Insufficient source family for separated development bands")
        counts = [int(available * ratio) for ratio in (0.6, 0.15, 0.1)]
        counts.append(available - sum(counts))
        offset = 0
        for split, count in zip(("train", "validation", "calibration", "test"), counts, strict=True):
            result[(family, split)] = ordered[offset : offset + count]
            offset += count + guard
    return result


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError("Use a new generation directory; existing data is preserved")
    acquisition = json.loads((archive.parent / "acquisition.json").read_text())
    if acquisition.get("state") != "verified_download" or archive.stat().st_size != acquisition["bytes"]:
        raise ValueError("Verified acquisition record is required")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != acquisition["sha256"]:
            raise ValueError("Archive checksum changed")
    samples, rejected, hashes = [], [], set()
    audit_counts, strata = Counter(), Counter()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        images = [name for name in names if name.endswith(".jpg")]
        bands = filename_bands(images)
        labels_by_image = {}
        for image in images:
            label_path = image.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
            if bundle.getinfo(label_path).file_size > 1_000_000:
                raise ValueError("Unexpected label file size")
            try:
                boxes, labels = parse_labels(bundle.read(label_path).decode("utf-8"))
                labels_by_image[image] = (boxes, labels)
                audit_counts.update(str(label) for label in labels)
                strata[str(sorted(set(labels)))] += 1
            except ValueError as exc:
                rejected.append({"image": image, "reason": str(exc)})
        output.mkdir(parents=True)
        (output / "images").mkdir()
        for (family, split), members in bands.items():
            # Seeded, stratified sampling within each disjoint filename band.
            buckets = defaultdict(list)
            for name in members:
                if name in labels_by_image:
                    buckets[tuple(sorted(set(labels_by_image[name][1])))].append(name)
            for members_in_class in buckets.values():
                members_in_class.sort(
                    key=lambda name: hashlib.sha256(("dfire-v1:" + name).encode()).hexdigest()
                )
            limit = 128 if split == "train" else 32
            selected = []
            while len(selected) < limit and any(buckets.values()):
                for key in sorted(buckets):
                    if buckets[key] and len(selected) < limit:
                        selected.append(buckets[key].pop())
            for name in selected:
                if bundle.getinfo(name).file_size > 10_000_000:
                    raise ValueError("Unexpected image size")
                content = bundle.read(name)
                sha = hashlib.sha256(content).hexdigest()
                if sha in hashes:
                    rejected.append({"image": name, "reason": "Duplicate selected image content"})
                    continue
                if cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) is None:
                    rejected.append({"image": name, "reason": "Image decode failed"})
                    continue
                hashes.add(sha)
                relative = f"images/{sha}.jpg"
                (output / relative).write_bytes(content)
                boxes, labels = labels_by_image[name]
                samples.append(
                    {
                        "image": relative,
                        "sha256": sha,
                        "split": split,
                        "group": f"dfire-{family}-development-band-{split}",
                        "boxes": boxes,
                        "labels": labels,
                        "source_member": name,
                    }
                )
    manifest = {
        "schema_version": 1,
        "name": "D-Fire bounded development generation 1",
        "task": "detection",
        "domain": "fire-smoke-development",
        "synthetic": False,
        "classes": ["smoke", "fire"],
        "capability_ids": ["fire_smoke"],
        "license": acquisition["license"],
        "samples": samples,
        "provenance": {
            "source": acquisition["page"],
            "archive_sha256": acquisition["sha256"],
            "source_independence_verified": False,
            "group_method": "Filename bands per source family, 32 omitted images between bands",
            "limitation": "No camera/session identities; correlated scenes may remain. Development metrics only. Independent site acceptance required.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "archive_images": len(images),
        "archive_boxes_by_class": dict(audit_counts),
        "archive_image_label_strata": dict(strata),
        "selected_images": len(samples),
        "split_counts": dict(Counter(sample["split"] for sample in samples)),
        "rejected": rejected,
        "source_independence_verified": False,
        "production_acceptance_eligible": False,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_detection_manifest(output / "manifest.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))
