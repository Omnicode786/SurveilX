"""Prepared-image integrity for optimizer resume; verification never rewrites data."""

import json
from pathlib import Path

from training.datasets import digest


def inventory(directory):
    directory = Path(directory).resolve()
    if not (directory / "dataset.yaml").is_file():
        raise ValueError("Prepared dataset.yaml is missing")
    if not (directory / "images").is_dir() or not (directory / "labels").is_dir():
        raise ValueError("Prepared image/label directories are missing")
    files = [directory / "dataset.yaml", *sorted((directory / "images").rglob("*")),
             *sorted((directory / "labels").rglob("*.txt"))]
    result = {}
    for file in files:
        if not file.is_file():
            continue
        if not file.resolve().is_relative_to(directory):
            raise ValueError("Prepared asset leaves its dataset directory")
        result[file.relative_to(directory).as_posix()] = digest(file)
    return result


def bind_prepared_data(directory, manifest, size, preprocessing, policy_file, sources):
    directory = Path(directory)
    record = {"schema_version": 1, "dataset_sha256": digest(manifest), "image_size": size,
        "preprocessing": preprocessing, "policy_sha256": digest(policy_file),
        "sources": {p.name: digest(p) for p in sources}, "files": inventory(directory)}
    (directory / "prepared-integrity.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def verified_prepared_data(directory, manifest, size, preprocessing, policy_file, sources):
    directory = Path(directory)
    file = directory / "prepared-integrity.json"
    if not file.exists():
        raise ValueError("Legacy prepared data has no integrity record; preserve it and initialize weights in a new version")
    record = json.loads(file.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "dataset_sha256": digest(manifest), "image_size": size,
        "preprocessing": preprocessing, "policy_sha256": digest(policy_file),
        "sources": {p.name: digest(p) for p in sources}, "files": inventory(directory)}
    if record != expected:
        raise ValueError("Prepared images, labels, policy, source or geometry changed; optimizer resume rejected")
    return directory / "dataset.yaml"
