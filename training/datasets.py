import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "calibration", "test")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    from surveilx.task_profiles import validate_profiles

    validate_profiles(manifest)
    if manifest.get("task", "event") not in {"event", "detection"}:
        raise ValueError("Unsupported task; detection and video-event classification are supported")
    if manifest.get("task") == "detection":
        from training.detection_pipeline import validate_detection_manifest

        manifest, counts, _ = validate_detection_manifest(path)
        return manifest, counts
    if manifest.get("schema_version") != 1 or len(manifest.get("classes", [])) < 2:
        raise ValueError("Manifest needs schema_version 1 and at least two classes")
    if not manifest.get("license") or not manifest.get("domain"):
        raise ValueError("Dataset license and domain are required")
    contract = manifest.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
    scene = manifest.get("representation") == "scene_clip"
    if manifest.get("representation", "entity_clip") not in {"entity_clip", "scene_clip"}:
        raise ValueError("Unknown event representation")
    frames, entities, size = (contract.get(key) for key in ("frames", "entities", "image_size"))
    if not all(isinstance(x, int) for x in (frames, entities, size)) or not (
        2 <= frames <= 64 and (entities == 0 if scene else 1 <= entities <= 32) and 32 <= size <= 512
    ):
        raise ValueError("Event contract needs frames 2..64, entities 1..32 and image_size 32..512")
    if scene and not 0.5 <= contract.get("clip_seconds", 0) <= 30:
        raise ValueError("Scene clips require a fixed duration between 0.5 and 30 seconds")
    seen_sources = {}
    seen_groups, seen_hashes, counts = {}, {}, {split: 0 for split in SPLITS}
    for sample in manifest["samples"]:
        split, group = sample["split"], sample["group"]
        if split not in SPLITS or not group:
            raise ValueError("Invalid split/group")
        if group in seen_groups and seen_groups[group] != split:
            raise ValueError("Camera/session group leaks across splits")
        seen_groups[group] = split
        source = sample.get("source_video_sha256")
        if source:
            if source in seen_sources and seen_sources[source] != split:
                raise ValueError("Source video leaks across splits")
            seen_sources[source] = split
        file = (path.parent / sample["file"]).resolve()
        if not file.is_relative_to(path.parent) or file.suffix != ".npz":
            raise ValueError("Sample must be an NPZ within dataset directory")
        checksum = digest(file)
        if sample.get("sha256") and sample["sha256"] != checksum:
            raise ValueError("Sample checksum mismatch")
        if checksum in seen_hashes and seen_hashes[checksum] != split:
            raise ValueError("Identical sample leaks across splits")
        seen_hashes[checksum] = split
        with np.load(file, allow_pickle=False) as data:
            clip, context = data["clip"], data["context"]
            boxes = np.empty((frames, 0, 4), dtype=np.float32) if scene else data["boxes"]
            if (
                clip.shape != (frames, 3, size, size)
                or boxes.shape != (frames, entities, 4)
                or context.shape != (4,)
            ):
                raise ValueError("Sample dimensions disagree with the event input_contract")
            if not all(np.isfinite(x).all() for x in (clip, boxes, context)):
                raise ValueError("Non-finite sample")
            if clip.min() < 0 or clip.max() > 1 or (boxes.size and (boxes.min() < 0 or boxes.max() > 1)):
                raise ValueError("Clip and boxes must be normalized to [0,1]")
            if np.any(boxes[..., 2:] <= boxes[..., :2]):
                raise ValueError("Invalid box geometry")
        if not 0 <= sample["label"] < len(manifest["classes"]):
            raise ValueError("Class outside taxonomy")
        counts[split] += 1
    if any(count < 10 for count in counts.values()):
        raise ValueError("Each independent split requires at least 10 samples")
    return manifest, counts


def generate(directory, count=160, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("Dataset exists; use a new version directory")
    if count < 80:
        raise ValueError("Generate at least 80 samples")
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(count):
        split = SPLITS[min(3, int(index / count * 4))]
        label = index % 2
        clip = rng.uniform(0, 0.12, (8, 3, 64, 64)).astype("float32")
        boxes = np.zeros((8, 2, 4), dtype="float32")
        for t in range(8):
            x = (8 + t * 3 if label else 8 + t // 3) + int(rng.integers(0, 3))
            for entity, (cx, cy) in enumerate([(x, 20), (43, 35)]):
                clip[t, 1, cy : cy + 12, cx : cx + 8] = 0.85
                boxes[t, entity] = [cx / 64, cy / 64, (cx + 8) / 64, (cy + 12) / 64]
        file = f"sample-{index:05d}.npz"
        np.savez_compressed(
            directory / file, clip=clip, boxes=boxes, context=np.array([1, 0, 0, 0], dtype="float32")
        )
        samples.append(
            {
                "file": file,
                "split": split,
                "group": f"generated-{seed}-{index}",
                "label": label,
                "sha256": digest(directory / file),
            }
        )
    manifest = {
        "schema_version": 1,
        "name": directory.name,
        "synthetic": True,
        "seed": seed,
        "license": "Generated locally for software verification",
        "domain": "synthetic-motion",
        "classes": ["slow_motion", "fast_motion"],
        "samples": samples,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory / "manifest.json"
