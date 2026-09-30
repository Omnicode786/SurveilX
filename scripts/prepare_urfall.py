"""Acquire a bounded, sequence-disjoint UR Fall RGB development subset."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from scripts.remote_zip import RemoteZip
from training.datasets import SPLITS, digest, validate_manifest


BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
CLASSES = ["normal", "fall"]
CONTRACT = {"frames": 8, "entities": 0, "image_size": 96, "clip_seconds": 1.5}


def write_manifest(output, samples):
    manifest = {
        "schema_version": 1,
        "name": output.name,
        "task": "event",
        "representation": "scene_clip",
        "input_contract": CONTRACT,
        "domain": "indoor-fall-development",
        "classes": CLASSES,
        "license": "UR Fall CC BY-NC-SA 4.0; noncommercial academic development only",
        "synthetic": False,
        "capability_ids": ["fall"],
        "samples": samples,
        "provenance": {
            "publisher": "University of Rzeszow UR Fall Detection Dataset",
            "publisher_page": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
            "label_scope": "whole_clip_sequence_category; fall transition only localizes positive clips",
            "source_independence_verified": False,
            "reason": "Sequence groups are disjoint, but participant/site identities are not published",
            "rgb_depth_timing_note": "Publisher posture labels refer to depth frames; they are not copied as RGB frame labels",
        },
    }
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(pending)
    pending.replace(output / "manifest.json")
    return output / "manifest.json"


def fetch_rows(name):
    request = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "SurveilX-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return [line.split(",") for line in response.read().decode("utf-8").splitlines() if line]


def source_plan():
    fall_rows = fetch_rows("urfall-cam0-falls.csv")
    counts, transitions = {}, {}
    for row in fall_rows:
        sequence, frame, label = row[0], int(row[1]), int(row[2])
        counts[sequence] = max(counts.get(sequence, 0), frame)
        if label == 0:
            transitions.setdefault(sequence, []).append(frame)
    adl_rows = fetch_rows("urfall-cam0-adls.csv")
    for row in adl_rows:
        sequence, frame = row[0], int(row[1])
        counts[sequence] = max(counts.get(sequence, 0), frame)
    plan = []
    for split_index, split in enumerate(SPLITS):
        for kind, label in (("fall", 1), ("adl", 0)):
            for number in range(split_index * 5 + 1, split_index * 5 + 6):
                sequence = f"{kind}-{number:02d}"
                if kind == "fall":
                    if not transitions.get(sequence):
                        raise ValueError(f"Publisher transition is missing for {sequence}")
                    center = int(np.median(transitions[sequence]))
                else:
                    center = counts[sequence] // 2
                # 1.5 seconds at the publisher's 30 FPS cadence, inclusive endpoints.
                start = max(1, min(center - 22, counts[sequence] - 45))
                indices = np.rint(np.linspace(start, start + 45, CONTRACT["frames"])).astype(int).tolist()
                plan.append((sequence, split, label, indices, counts[sequence]))
    return plan


def _prepare(output):
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        validate_manifest(manifest_path)
        return manifest_path
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise ValueError("At least 2 GiB free space is required for safe acquisition")
    status_path = output / "acquisition.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"samples": {}, "transferred": 0}
    samples = []
    for sequence, split, label, indices, source_frames in source_plan():
        # Reload after every source so a resumed importer never works from a stale journal snapshot.
        if status_path.exists():
            status = json.loads(status_path.read_text())
        name = f"{sequence}.npz"
        path = output / name
        prior = status["samples"].get(sequence)
        if path.exists() and prior and digest(path) == prior.get("sha256"):
            samples.append(prior["sample"])
            continue
        url = f"{BASE}/{sequence}-cam0-rgb.zip"
        remote = RemoteZip(url, budget=64 * 1024**2)
        frames, members = [], []
        with ZipFile(remote) as archive:
            lookup = {Path(item.filename).name: item for item in archive.infolist() if not item.is_dir()}
            for frame in indices:
                member_name = f"{sequence}-cam0-rgb-{frame:03d}.png"
                if member_name not in lookup:
                    raise ValueError(f"Missing reviewed RGB frame {member_name}")
                raw = archive.read(lookup[member_name])  # ZipFile verifies each member CRC.
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Invalid RGB frame {member_name}")
                image = cv2.cvtColor(cv2.resize(image, (96, 96)), cv2.COLOR_BGR2RGB)
                frames.append(image.transpose(2, 0, 1).astype(np.float32) / 255)
                members.append({"name": member_name, "sha256": hashlib.sha256(raw).hexdigest()})
        np.savez_compressed(path, clip=np.stack(frames), context=np.zeros(4, dtype=np.float32))
        sample = {
            "file": name,
            "sha256": digest(path),
            "label": label,
            "split": split,
            "group": sequence,
            "source_archive_etag": remote.etag,
            "source_archive_size": remote.size,
            "source_frame_range": [indices[0], indices[-1]],
        }
        status["transferred"] += remote.transferred
        status["samples"][sequence] = {
            "sample": sample,
            "source_url": url,
            "source_frames": source_frames,
            "members": members,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        samples.append(sample)
    return write_manifest(output, samples)


def prepare(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f"{output.name}.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.close(descriptor)
    try:
        return _prepare(output)
    finally:
        lock.unlink(missing_ok=True)


def reconcile(output):
    """Re-journal intact derived NPZs after an interrupted concurrent importer."""
    output = Path(output).resolve()
    status_path = output / "acquisition.json"
    status = json.loads(status_path.read_text())
    repaired = []
    for sequence, entry in status["samples"].items():
        path = output / entry["sample"]["file"]
        with np.load(path, allow_pickle=False) as data:
            clip, context = data["clip"], data["context"]
            if (
                clip.shape != (CONTRACT["frames"], 3, CONTRACT["image_size"], CONTRACT["image_size"])
                or context.shape != (4,)
                or not np.isfinite(clip).all()
                or clip.min() < 0
                or clip.max() > 1
            ):
                raise ValueError(f"Cannot reconcile invalid derived sample {sequence}")
        checksum = digest(path)
        if checksum != entry["sample"].get("sha256"):
            entry["sample"]["sha256"] = checksum
            repaired.append(sequence)
    status["reconciliation"] = {
        "reason": "Recovered intact derived samples after an interrupted concurrent importer",
        "updated_samples": repaired,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return repaired


def finalize(output):
    """Publish a complete reconciled journal without any network or source-file mutation."""
    output = Path(output).resolve()
    status = json.loads((output / "acquisition.json").read_text())
    expected = {
        sequence: (split, label, [indices[0], indices[-1]])
        for sequence, split, label, indices, _ in source_plan()
    }
    if set(status["samples"]) != set(expected):
        raise ValueError("Acquisition journal does not contain the complete 40-sequence plan")
    samples = []
    for sequence, (split, label, frame_range) in expected.items():
        sample = status["samples"][sequence]["sample"]
        if (
            sample.get("group") != sequence
            or sample.get("split") != split
            or sample.get("label") != label
            or sample.get("source_frame_range") != frame_range
            or digest(output / sample["file"]) != sample.get("sha256")
        ):
            raise ValueError(f"Acquisition journal mismatch for {sequence}")
        samples.append(sample)
    return write_manifest(output, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="data/datasets/urfall-development-v1")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.reconcile and args.finalize:
        parser.error("Choose at most one recovery action")
    print(finalize(args.output) if args.finalize else reconcile(args.output) if args.reconcile else prepare(args.output))
