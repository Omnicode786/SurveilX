"""Acquire a bounded, paired-camera AIRTLab fighting development dataset.

Publisher video labels are weak labels for the center crop, not reviewed temporal
annotations. These staged, single-room samples cannot establish field reliability.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request

from scripts.prepare_scene_event_directory import video_duration
from training.datasets import digest, validate_manifest
from training.import_scene import import_scene


COMMIT = "1f7747e104301ccaa82ef5a2f6804b51ced1c398"
REPOSITORY = "airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos"
PREFIX = "violence-detection-dataset/"
RIGHTS = "AIRTLab: research and educational purposes; see publisher readme.md Dataset Release Agreement"


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(entry, root, opener=urllib.request.urlopen):
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
        raise ValueError("Unsafe publisher path")
    target = Path(root).joinpath(*relative.parts)
    if target.exists():
        data = target.read_bytes()
    else:
        if shutil.disk_usage(root).free < entry["size"] + 2 * 1024**3:
            raise ValueError("Acquisition requires a 2 GiB free-space reserve")
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{relative}"
        with opener(url, timeout=90) as response:
            data = response.read(entry["size"] + 1)
    if len(data) != entry["size"] or git_blob_sha(data) != entry["sha"]:
        raise ValueError(f"Publisher blob integrity failed: {relative}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
    return target


def actions(text):
    rows = csv.reader(io.StringIO(text), delimiter=";")
    next(rows)
    return {row[0].strip(): set(row[1].strip().split(",")) for row in rows if len(row) == 2}


def plan(entries, normal, violent, seed=42):
    """Freeze 40 action groups per class; paired views always share a split."""
    lookup = {item["path"]: item for item in entries}
    result = []
    for folder, label, candidates in (
        ("non-violent", "normal", list(normal)),
        (
            "violent",
            "fighting",
            [
                name
                for name, tags in violent.items()
                if "fight" in tags and not tags.intersection({"gunshot", "stab"})
            ],
        ),
    ):
        candidates.sort(key=lambda name: hashlib.sha256(f"{seed}/{folder}/{name}".encode()).hexdigest())
        if len(candidates) < 40:
            raise ValueError("Forty source actions per class are required")
        for index, name in enumerate(candidates[:40]):
            split = (
                "train"
                if index < 25
                else "validation"
                if index < 30
                else "calibration"
                if index < 35
                else "test"
            )
            for camera in ("cam1", "cam2"):
                path = f"{PREFIX}{folder}/{camera}/{name}"
                item = lookup[path]
                result.append(
                    {
                        "entry": item,
                        "label": label,
                        "split": split,
                        "group": f"airtlab-{folder}-{Path(name).stem}",
                    }
                )
    return result


def prepare(root, output, max_bytes=650 * 1024**2):
    root, output = Path(root).resolve(), Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "inventory.json"
    if not inventory_path.exists():
        with urllib.request.urlopen(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1", timeout=60
        ) as response:
            inventory_path.write_bytes(response.read(2 * 1024**2))
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("sha") != COMMIT or inventory.get("truncated"):
        raise ValueError("Expected the complete pinned publisher inventory")
    entries = inventory["tree"]
    lookup = {item["path"]: item for item in entries}
    readme = fetch(lookup["readme.md"], root)
    normal = fetch(lookup[PREFIX + "nonviolent-action-classes.csv"], root)
    violent = fetch(lookup[PREFIX + "violent-action-classes.csv"], root)
    selected = plan(entries, actions(normal.read_text()), actions(violent.read_text()))
    total = sum(item["entry"]["size"] for item in selected)
    if total > max_bytes:
        raise ValueError(f"Frozen subset exceeds acquisition budget: {total} > {max_bytes}")
    config = {
        "publisher_commit": COMMIT,
        "inventory_sha256": digest(inventory_path),
        "selected_bytes": total,
        "selected": selected,
        "label_scope": "inherited_video_label",
    }
    plan_path = root / "fighting-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise ValueError("Frozen acquisition plan changed; use a new acquisition directory")
    plan_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if (output / "manifest.json").exists():
        manifest, _ = validate_manifest(output / "manifest.json")
        if manifest.get("provenance", {}).get("acquisition_plan_sha256") != digest(plan_path):
            raise ValueError("Existing dataset does not match the pinned acquisition")
        return output / "manifest.json"
    if output.exists():
        raise ValueError("Partial import preserved; inspect before choosing a new output directory")
    clips, records = [], []
    journal_path = root / "acquisition.json"
    for index, item in enumerate(selected):
        path = fetch(item["entry"], root)
        duration = video_duration(path)
        # 1.5 seconds fits the publisher's shortest two-second videos.
        if duration < 1.5:
            raise ValueError(f"Source is shorter than the fixed contract: {path.name}")
        start = max(0.0, (duration - 1.5) / 2)
        clips.append(
            {
                "video": str(path.relative_to(root)),
                "label": item["label"],
                "split": item["split"],
                "group": item["group"],
                "start_seconds": start,
                "end_seconds": start + 1.5,
            }
        )
        records.append(
            {
                "file": item["entry"]["path"],
                "sha256": digest(path),
                "group": item["group"],
                "split": item["split"],
            }
        )
        journal = {
            "state": "acquiring",
            "complete": index + 1,
            "total": len(selected),
            "plan_sha256": digest(plan_path),
            "records": records,
        }
        pending = journal_path.with_suffix(".tmp")
        pending.write_text(json.dumps(journal, indent=2), encoding="utf-8")
        pending.replace(journal_path)
        print(json.dumps({"acquired": index + 1, "total": len(selected)}), flush=True)
    descriptor = {
        "classes": ["normal", "fighting"],
        "domain": "general",
        "license": RIGHTS,
        "source_root": str(root),
        "input_contract": {"frames": 8, "entities": 0, "image_size": 96, "clip_seconds": 1.5},
        "synthetic": False,
        "source_independence_verified": False,
        "capability_ids": ["fighting"],
        "label_scope": "inherited_video_label",
        "provenance": {
            "publisher_url": f"https://github.com/{REPOSITORY}",
            "publisher_commit": COMMIT,
            "acquisition_plan_sha256": digest(plan_path),
            "rights_sha256": digest(readme),
            "staged": True,
            "single_room": True,
            "paired_camera_grouping": True,
            "temporal_annotations_reviewed": False,
            "limitations": "Center crops inherit video labels; actors/sites are not independent across splits. Research/education only.",
        },
        "clips": clips,
    }
    descriptor_path = root / "fighting-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    artifact = import_scene(descriptor_path, output)
    journal.update(state="completed", manifest=str(artifact), manifest_sha256=digest(artifact))
    journal_path.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    print(prepare(args.root, args.output))
