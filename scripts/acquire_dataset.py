"""Acquire and audit the official Penn-Fudan pedestrian development dataset.

Run from the repository root with its Python environment. Data stays out of Git.
The manifest uses normalized, half-open xyxy boxes and zero-based class IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np


SOURCE_PAGE = "https://www.cis.upenn.edu/~jshi/ped_html/"
SOURCE_URL = SOURCE_PAGE + "PennFudanPed.zip"
EXPECTED_SHA256 = "9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2"
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
MAX_EXTRACTED_BYTES = 120 * 1024 * 1024
SPLITS = ("train", "validation", "calibration", "test")
LICENSE = (
    "Copyright retained by authors/holders; copying is subject to their terms and most reposting requires "
    "explicit permission. No standard open-data license or commercial rights established. "
    "Downloaded for local research evaluation; do not redistribute without the required permission."
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(archive: Path) -> None:
    if archive.exists():
        if digest(archive) != EXPECTED_SHA256:
            raise ValueError(
                f"Archive checksum mismatch: {archive}; preserve it and use a new output directory"
            )
        return
    temporary = archive.with_suffix(".zip.part")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "SurveilX-dataset-acquisition/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            if source.url != SOURCE_URL:
                raise ValueError("Unexpected download redirect")
            total = 0
            while block := source.read(1024 * 1024):
                total += len(block)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Dataset download exceeds its size limit")
                target.write(block)
    except urllib.error.URLError:
        if os.name != "nt":
            raise
        # Windows' native TLS stack can build certificate issuer chains absent
        # from a Python OpenSSL installation. Both paths verify TLS certificates.
        escaped_path = str(temporary.resolve()).replace("'", "''")
        command = (
            "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; "
            f"Invoke-WebRequest -UseBasicParsing -Uri '{SOURCE_URL}' -OutFile '{escaped_path}'"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True, timeout=900
        )
    if temporary.stat().st_size > MAX_DOWNLOAD_BYTES or digest(temporary) != EXPECTED_SHA256:
        raise ValueError("Downloaded archive does not match the pinned official dataset checksum")
    temporary.replace(archive)


def extract_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 2000 or sum(entry.file_size for entry in entries) > MAX_EXTRACTED_BYTES:
            raise ValueError("Unexpected archive expansion size")
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or not name.parts
                or name.parts[0] != "PennFudanPed"
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            target = (root / Path(*name.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path leaves the output directory")
        # Validate the entire archive before creating any of its files.
        for entry in entries:
            target = root / Path(*PurePosixPath(entry.filename).parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def boxes_from_mask(mask_path: Path, width: int, height: int) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape[:2] != (height, width):
        raise ValueError(f"Invalid or mismatched instance mask: {mask_path}")
    if mask.ndim == 3:
        # OpenCV decodes a PNG palette into BGR. Keep each exact palette color
        # distinct, rather than converting to grayscale and merging instances.
        pixels = mask[..., :3].astype(np.uint32)
        mask = pixels[..., 0] + (pixels[..., 1] << 8) + (pixels[..., 2] << 16)
    boxes = []
    for instance in np.unique(mask):
        if instance == 0:
            continue
        ys, xs = np.where(mask == instance)
        boxes.append(
            [
                float(xs.min() / width),
                float(ys.min() / height),
                float((xs.max() + 1) / width),
                float((ys.max() + 1) / height),
            ]
        )
    return sorted(boxes)


def partition(images: list[Path]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Use contiguous filename bands, with one unused boundary image per split.

    This is an approximation: the publisher does not provide camera/session IDs.
    The guard images reduce immediate adjacency leakage, but cannot rule it out.
    """
    assignments, excluded = {}, []
    for campus in ("PennPed", "FudanPed"):
        members = sorted(image for image in images if image.stem.startswith(campus))
        usable = len(members) - 3
        counts = [int(usable * 0.60), int(usable * 0.15), int(usable * 0.10)]
        counts.append(usable - sum(counts))
        offset = 0
        for index, (split, count) in enumerate(zip(SPLITS, counts)):
            for image in members[offset : offset + count]:
                assignments[image.name] = (split, f"{campus.lower()}-filename-band-{split}")
            offset += count
            if index < len(SPLITS) - 1:
                excluded.append(members[offset].name)
                offset += 1
    return assignments, excluded


def acquire(destination: Path) -> dict:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provenance", {}).get("archive_sha256") != EXPECTED_SHA256:
            raise ValueError(
                "Existing manifest differs from this acquisition recipe; use a new output directory"
            )
        for sample in manifest["samples"]:
            image = (destination / sample["image"]).resolve()
            if not image.is_relative_to(destination) or digest(image) != sample["sha256"]:
                raise ValueError("Existing dataset image failed its provenance check")
        return manifest
    archive = destination / "PennFudanPed.zip"
    download(archive)
    extract_safely(archive, destination)
    dataset = destination / "PennFudanPed"
    images = sorted((dataset / "PNGImages").glob("*.png"))
    if len(images) != 170:
        raise ValueError(f"Expected the official 170 images; found {len(images)}")
    assignments, excluded = partition(images)
    samples, all_instances, seen_hashes = [], 0, {}
    for image in images:
        decoded = cv2.imread(str(image))
        if decoded is None:
            raise ValueError(f"Unreadable image: {image}")
        height, width = decoded.shape[:2]
        boxes = boxes_from_mask(dataset / "PedMasks" / f"{image.stem}_mask.png", width, height)
        annotation = (dataset / "Annotation" / f"{image.stem}.txt").read_text(encoding="utf-8")
        declared = re.search(r"Objects with ground truth\s*:\s*(\d+)", annotation)
        if declared is None or int(declared.group(1)) != len(boxes):
            raise ValueError(f"Mask instances disagree with original annotation: {image.name}")
        all_instances += len(boxes)
        if image.name not in assignments:
            continue
        split, group = assignments[image.name]
        checksum = digest(image)
        if checksum in seen_hashes and seen_hashes[checksum] != split:
            raise ValueError("Identical images cross dataset splits")
        seen_hashes[checksum] = split
        sample = {
            "image": image.relative_to(destination).as_posix(),
            "boxes": boxes,
            "labels": [0] * len(boxes),
            "split": split,
            "group": group,
            "sha256": checksum,
            "width": width,
            "height": height,
        }
        samples.append(sample)
        yolo_image = destination / "yolo" / "images" / split / image.name
        yolo_label = destination / "yolo" / "labels" / split / f"{image.stem}.txt"
        yolo_image.parent.mkdir(parents=True, exist_ok=True)
        yolo_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, yolo_image)
        yolo_label.write_text(
            "".join(
                f"0 {(x1 + x2) / 2:.9f} {(y1 + y2) / 2:.9f} {x2 - x1:.9f} {y2 - y1:.9f}\n"
                for x1, y1, x2, y2 in boxes
            ),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "task": "detection",
        "name": "pennfudan-person-v1",
        "classes": ["person"],
        "domain": "pedestrian",
        "license": LICENSE,
        "synthetic": False,
        "provenance": {
            "source_page": SOURCE_PAGE,
            "archive_url": SOURCE_URL,
            "archive_sha256": digest(archive),
            "archive_bytes": archive.stat().st_size,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "publisher_image_count": 170,
            "publisher_instance_count": 345,
            "decoded_instance_count": all_instances,
            "license_status": "restricted-copyright-notice-no-standard-license-grant",
            "rights_notice_file": "PennFudanPed/readme.txt",
            "rights_notice_sha256": digest(dataset / "readme.txt"),
            "annotation_count_note": "Archive readme explains 78 additional small/occluded pedestrians "
            "beyond the website's original 345; released masks contain 423 instances.",
            "upstream_reference": "Object Detection Combining Recognition and Segmentation, ACCV 2007",
        },
        "split_policy": {
            "method": "contiguous-filename-bands-per-campus-with-boundary-guards",
            "nominal_fractions": {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15},
            "guard_images_excluded": excluded,
            "grouping_limitation": "Filename bands are proxies; true camera/session identities are unavailable. "
            "Distinct files and hashes do not guarantee scene-independent evaluation.",
        },
        "limitations": [
            "Small development dataset with upright pedestrians only; not a broad surveillance benchmark.",
            "No labeled negative images, actions, violence, weapons, night scenes, or camera timing metadata.",
            "Calibration on this small set does not establish deployment accuracy or superiority to YOLO.",
        ],
        "samples": samples,
    }
    # JSON-quoted paths are valid YAML and avoid Windows backslash escaping issues.
    yolo_root = json.dumps((destination / "yolo").as_posix())
    (destination / "dataset.yaml").write_text(
        f"path: {yolo_root}\ntrain: images/train\nval: images/validation\ntest: images/test\n"
        "names:\n  0: person\n# Calibration is deliberately excluded from train/val/test.\n",
        encoding="utf-8",
    )
    calibration_path = destination / "calibration.yaml"
    calibration_path.write_text(
        f"path: {yolo_root}\ntrain: images/calibration\nval: images/calibration\n"
        "names:\n  0: person\n# Only for post-training calibration/export. Never use for training/model selection.\n",
        encoding="utf-8",
    )
    (destination / "provenance.json").write_text(
        json.dumps(
            {key: manifest[key] for key in ("license", "provenance", "split_policy", "limitations")}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/datasets/pennfudan"))
    arguments = parser.parse_args()
    acquired = acquire(arguments.output)
    print(
        json.dumps(
            {
                "manifest": str((arguments.output / "manifest.json").resolve()),
                "images": dict(Counter(sample["split"] for sample in acquired["samples"])),
                "instances": dict(
                    Counter(
                        split
                        for sample in acquired["samples"]
                        for split in [sample["split"]]
                        for _ in sample["boxes"]
                    )
                ),
                "archive_sha256": acquired["provenance"]["archive_sha256"],
            },
            indent=2,
        )
    )
