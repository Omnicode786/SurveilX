"""Expand only D-Fire training bands; preserve every original held-out record."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import cv2
import numpy as np

from scripts.prepare_dfire import filename_bands, parse_labels
from training.datasets import digest
from training.detection_pipeline import validate_detection_manifest


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def expand(archive, base, output, per_family=300):
    archive, base, output = map(Path, (archive, base, output))
    if not 1 <= per_family <= 500:
        raise ValueError("Bounded additions per family must be 1..500")
    if output.exists():
        raise ValueError("Preserve existing directories; choose a new dataset version")
    original, _, _ = validate_detection_manifest(base / "manifest.json")
    acquisition = json.loads((archive.parent / "acquisition.json").read_text())
    if acquisition["state"] != "verified_download" or digest(archive) != acquisition["sha256"]:
        raise ValueError("Frozen archive checksum mismatch")
    if original["provenance"]["archive_sha256"] != acquisition["sha256"]:
        raise ValueError("Parent uses another source archive")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise ValueError("Maintain at least 2 GiB disk reserve")
    used_names = {sample["source_member"] for sample in original["samples"]}
    used_hashes = {sample["sha256"] for sample in original["samples"]}
    additions = []
    with zipfile.ZipFile(archive) as bundle:
        images = [name for name in bundle.namelist() if name.endswith(".jpg")]
        bands = filename_bands(images)
        for (family, split), names in bands.items():
            if split != "train":
                continue
            strata = defaultdict(list)
            for name in names:
                if name in used_names:
                    continue
                label_path = name.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
                if bundle.getinfo(label_path).file_size > 1_000_000:
                    raise ValueError("Unexpected annotation size")
                try:
                    boxes, labels = parse_labels(bundle.read(label_path).decode("utf-8"))
                except ValueError:
                    continue
                strata[tuple(sorted(set(labels)))].append((name, boxes, labels))
            for members in strata.values():
                members.sort(
                    key=lambda row: hashlib.sha256(("dfire-expansion-g2:" + row[0]).encode()).hexdigest()
                )
            count = 0
            while count < per_family and any(strata.values()):
                for key in sorted(strata):
                    if not strata[key] or count >= per_family:
                        continue
                    name, boxes, labels = strata[key].pop()
                    if bundle.getinfo(name).file_size > 10_000_000:
                        raise ValueError("Unexpected image size")
                    content = bundle.read(name)
                    source_hash = hashlib.sha256(content).hexdigest()
                    if source_hash in used_hashes:
                        continue
                    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    used_hashes.add(source_hash)
                    additions.append(
                        {
                            "source_member": name,
                            "source_sha256": source_hash,
                            "split": "train",
                            "group": f"dfire-{family}-development-band-train",
                            "boxes": boxes,
                            "labels": labels,
                        }
                    )
                    count += 1
        estimate = sum((base / sample["image"]).stat().st_size for sample in original["samples"])
        estimate += sum(bundle.getinfo(sample["source_member"]).file_size for sample in additions)
        if shutil.disk_usage(output.parent).free < estimate + 2 * 1024**3:
            raise ValueError("Expansion would breach disk reserve")
        output.mkdir()
        (output / "images").mkdir()
        journal = {
            "state": "copying",
            "parent_manifest_sha256": digest(base / "manifest.json"),
            "archive_sha256": acquisition["sha256"],
            "per_family": per_family,
            "additions": additions,
            "created": time.time(),
        }
        save(output / "acquisition.json", journal)
        for sample in original["samples"]:
            shutil.copy2(base / sample["image"], output / sample["image"])
        for index, sample in enumerate(additions):
            content = bundle.read(sample["source_member"])
            if hashlib.sha256(content).hexdigest() != sample["source_sha256"]:
                raise ValueError("Source member changed during preparation")
            relative = f"images/{sample['source_sha256']}.jpg"
            (output / relative).write_bytes(content)
            sample.update(image=relative, sha256=sample["source_sha256"])
            if index % 100 == 0:
                journal["copied_additions"] = index + 1
                save(output / "acquisition.json", journal)
        manifest = {
            **original,
            "name": "D-Fire train-only expanded development generation 2",
            "samples": [*original["samples"], *additions],
            "provenance": {
                **original["provenance"],
                "parent_manifest_sha256": journal["parent_manifest_sha256"],
                "training_expansion": len(additions),
                "heldout_records_unchanged": True,
            },
        }
        save(output / "manifest.json", manifest)
        _, counts, _ = validate_detection_manifest(output / "manifest.json")
        before = [s for s in original["samples"] if s["split"] != "train"]
        after = [s for s in manifest["samples"] if s["split"] != "train"]
        if before != after:
            raise ValueError("Held-out records changed")
        journal.update(
            state="completed",
            counts=counts,
            manifest_sha256=digest(output / "manifest.json"),
            heldout_records_unchanged=True,
            finished=time.time(),
        )
        save(output / "acquisition.json", journal)
        return {
            key: journal[key] for key in ("state", "counts", "manifest_sha256", "heldout_records_unchanged")
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("base")
    parser.add_argument("output")
    parser.add_argument("--per-family", type=int, default=300)
    print(json.dumps(expand(**vars(parser.parse_args())), indent=2))
