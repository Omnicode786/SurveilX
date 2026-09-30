"""Import explicitly labeled fixed-duration video clips without inventing entity annotations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.datasets import digest, validate_manifest, SPLITS


def import_scene(descriptor, output):
    descriptor, output = Path(descriptor).resolve(), Path(output).resolve()
    spec = json.loads(descriptor.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Choose a new dataset directory")
    classes = spec["classes"]
    contract = spec.get("input_contract", {"frames": 8, "entities": 0, "image_size": 64, "clip_seconds": 2})
    frames, size, duration = contract["frames"], contract["image_size"], contract["clip_seconds"]
    if not (2 <= frames <= 64 and 32 <= size <= 512 and 0.5 <= duration <= 30 and contract["entities"] == 0):
        raise ValueError("Invalid scene clip contract")
    if (
        not spec.get("license")
        or not spec.get("domain")
        or len(classes) < 2
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Distinct classes, domain and rights record required")
    source_root = (descriptor.parent / spec.get("source_root", ".")).resolve()
    sources, groups, prepared = {}, {}, []
    for item in spec["clips"]:
        source = (source_root / item["video"]).resolve()
        if not source.is_relative_to(source_root) or not source.is_file():
            raise ValueError("Source video is missing or outside source_root")
        start, end = float(item["start_seconds"]), float(item["end_seconds"])
        if not np.isfinite([start, end]).all() or start < 0 or abs(end - start - duration) > 0.001:
            raise ValueError("Every clip needs the configured fixed duration")
        split, group = item["split"], item["group"]
        if split not in SPLITS or not group or item["label"] not in classes:
            raise ValueError("Explicit split, source group and clip label required")
        checksum = digest(source)
        if sources.get(checksum, split) != split or groups.get(group, split) != split:
            raise ValueError("Whole source videos and source groups must stay in one split")
        sources[checksum], groups[group] = split, split
        prepared.append((item, source, checksum, start, end))
    output.mkdir(parents=True)
    samples = []
    for index, (item, source, checksum, start, end) in enumerate(prepared):
        video = cv2.VideoCapture(str(source))
        try:
            fps, count = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
            if not np.isfinite([fps, count]).all() or fps <= 0 or end * fps >= count:
                raise ValueError(f"Clip outside decodable source video: {source.name}")
            clip = []
            indices = np.rint(np.linspace(start, end, frames) * fps).astype(int)
            if len(set(indices)) != frames:
                raise ValueError("Source frame rate cannot provide distinct observations")
            for frame_index in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, image = video.read()
                if not ok:
                    raise ValueError("Video decode failed within labeled interval")
                image = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
                clip.append(image.transpose(2, 0, 1).astype(np.float32) / 255)
        finally:
            video.release()
        name = f"clip-{index:06d}.npz"
        np.savez_compressed(output / name, clip=np.stack(clip), context=np.zeros(4, dtype=np.float32))
        samples.append(
            {
                "file": name,
                "sha256": digest(output / name),
                "label": classes.index(item["label"]),
                "split": item["split"],
                "group": item["group"],
                "source_video_sha256": checksum,
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    manifest = {
        "schema_version": 1,
        "name": output.name,
        "task": "event",
        "representation": "scene_clip",
        "input_contract": contract,
        "domain": spec["domain"],
        "classes": classes,
        "license": spec["license"],
        "synthetic": spec.get("synthetic", False),
        "capability_ids": spec.get("capability_ids", []),
        "samples": samples,
        "provenance": {
            "descriptor_sha256": digest(descriptor),
            "label_scope": "whole_clip_only",
            "source_independence_verified": spec.get("source_independence_verified", False),
        },
    }
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(pending)
    pending.replace(output / "manifest.json")
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor")
    parser.add_argument("output")
    args = parser.parse_args()
    print(import_scene(args.descriptor, args.output))
