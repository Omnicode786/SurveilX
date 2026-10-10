"""Prepare a deletion journal only for byte-reproducible YOLO materializations.

Does not delete anything, download, or train. Canonical datasets remain intact.
"""
import hashlib
import json
from pathlib import Path

import cv2
import yaml

from scripts.analyze_sva_datasets import cleanup_candidates, digest


def verify():
    cv2.setNumThreads(1)
    root = Path.cwd().resolve()
    journal_path = root / "reports/g4-cleanup-journal.json"
    if journal_path.exists():
        previous = json.loads(journal_path.read_text(encoding="utf-8-sig"))
        if previous.get("deleted") or any(row.get("deleted") for row in previous.get("directories", [])):
            raise ValueError("Preserve the completed/partial deletion journal; choose a new audit instead")
    report = {"scope": "Verified derived copies only; canonical datasets and model artifacts retained", "deleted": False, "directories": [], "retained": []}
    checked, encoded_hashes = {}, {}
    for candidate in cleanup_candidates(root)["candidates"]:
        directory = Path(candidate["directory"])
        if directory.parent.name == "dfire-g1-yolo-rai":
            report["retained"].append({"directory": str(directory), "reason": "Historical native-size validation image differs from both canonical bytes and current encoder/resize output. Preserve uncertain materialization."})
            continue
        print(f"Verifying {directory.parent.name}", flush=True)
        if not candidate["weights_sha256_matches"] or not directory.resolve().is_relative_to(root / "data/runs"):
            raise ValueError("Unsafe cleanup target")
        source = Path(candidate["source_manifest"])
        manifest = json.loads(source.read_text())
        artifact = json.loads((directory.parent / "manifest.json").read_text())
        size = artifact["config"]["image_size"]
        yaml_text = (directory / "dataset.yaml").read_text()
        if yaml.safe_load(yaml_text) != {"path": str(directory), "train": "images/train", "val": "images/validation",
                                       "test": "images/test", "names": dict(enumerate(manifest["classes"]))}:
            raise ValueError(f"Unexpected dataset configuration: {directory}")
        expected = {"dataset.yaml", "labels/train.cache", "labels/validation.cache"}
        inventory = []
        file_hashes = {}
        for index, sample in enumerate(manifest["samples"]):
            image = (source.parent / sample["image"]).resolve()
            if not image.is_relative_to(source.parent.resolve()):
                raise ValueError("Canonical asset escaped dataset")
            if image not in checked:
                checked[image] = digest(image)
            if checked[image] != sample["sha256"]:
                raise ValueError(f"Canonical source changed: {image}")
            relative = f"images/{sample['split']}/{index:06d}{image.suffix.lower()}"
            derived = directory / relative
            expected.add(relative)
            actual = digest(derived)
            file_hashes[relative] = actual
            if actual != checked[image]:
                key = (checked[image], size, image.suffix.lower())
                if key not in encoded_hashes:
                    ok, encoded = cv2.imencode(image.suffix.lower(), cv2.resize(cv2.imread(str(image)), (size, size)))
                    if not ok:
                        raise ValueError(f"Cannot re-encode {image}")
                    encoded_hashes[key] = hashlib.sha256(encoded.tobytes()).hexdigest()
                if encoded_hashes[key] != actual:
                    raise ValueError(f"Derived image is not byte reproducible: {derived}")
            label = f"labels/{sample['split']}/{index:06d}.txt"
            expected.add(label)
            lines = []
            for box, category in zip(sample["boxes"], sample["labels"], strict=True):
                x1, y1, x2, y2 = box
                lines.append(f"{category} {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}")
            if (directory / label).read_text() != "\n".join(lines):
                raise ValueError(f"Derived label modified: {label}")
        for path in directory.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Links cannot be cleaned recursively")
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                if relative not in expected:
                    raise ValueError(f"Unexpected file: {path}")
                inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hashes.get(relative) or digest(path)})
                if relative.startswith("images/"):
                    sample = manifest["samples"][int(path.stem)]
                    inventory[-1].update(source_image=sample["image"], source_sha256=sample["sha256"],
                        reconstruction="copy_original_bytes" if inventory[-1]["sha256"] == sample["sha256"] else
                        "cv2.resize(square_image_size)_then_cv2.imencode(original_extension)_default_quality")
        report["directories"].append({**candidate, "inventory": inventory, "verified_reproducible": True,
            "dataset_yaml_text": yaml_text,
            "rebuild": "Recreate images using per-file reconstruction and canonical source; YOLO labels from canonical boxes/classes; restore dataset_yaml_text. Disposable Ultralytics caches regenerate. No training needed.",
            "image_size": size})
        report["bytes_verified"] = sum(d["bytes"] for d in report["directories"])
        Path("reports/g4-cleanup-journal.json").write_text(json.dumps(report, indent=2))
    report["bytes_verified"] = sum(d["bytes"] for d in report["directories"])
    report["verification_complete"] = True
    report["yaml_configuration_verified"] = True
    Path("reports/g4-cleanup-journal.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"directories": len(report["directories"]), "bytes_verified": report["bytes_verified"]}))


if __name__ == "__main__":
    verify()
