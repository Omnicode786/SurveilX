"""Build a bounded scene-event dataset from user-provided split/class video folders."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from training.datasets import SPLITS, validate_manifest
from training.import_scene import import_scene


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def video_duration(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([fps, frames]).all() or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot read video timing: {path}")
        # Intervals are sampled inclusively; duration must end on a real frame.
        return (frames - 1) / fps
    finally:
        capture.release()


def clip_starts(duration, clip_seconds, max_clips):
    if duration + 1e-6 < clip_seconds:
        return []
    if max_clips <= 1:
        return [max(0.0, (duration - clip_seconds) / 2)]
    upper = max(0.0, duration - clip_seconds)
    return sorted(set((np.floor(np.linspace(0, upper, max_clips) * 1000) / 1000).tolist()))


def discover_clips(source_root, classes, contract, max_clips_per_video, max_clips_per_class_split):
    source_root = Path(source_root).resolve()
    clips = []
    for split in SPLITS:
        split_root = source_root / split
        if not split_root.is_dir():
            raise ValueError(f"Missing split directory: {split_root}")
        for label in classes:
            class_root = split_root / label
            if not class_root.is_dir():
                raise ValueError(f"Missing class directory: {class_root}")
            chosen = []
            for path in sorted(p for p in class_root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES):
                duration = video_duration(path)
                relative = path.relative_to(source_root).with_suffix("")
                group = "__".join(relative.parts)
                for start in clip_starts(duration, contract["clip_seconds"], max_clips_per_video):
                    chosen.append(
                        {
                            "video": str(path),
                            "start_seconds": start,
                            "end_seconds": round(start + contract["clip_seconds"], 3),
                            "split": split,
                            "group": group,
                            "label": label,
                        }
                    )
                    if len(chosen) >= max_clips_per_class_split:
                        break
                if len(chosen) >= max_clips_per_class_split:
                    break
            if not chosen:
                raise ValueError(f"No usable videos for {split}/{label}")
            clips.extend(chosen)
    return clips


def prepare(
    source_root,
    output,
    classes,
    domain,
    license_text,
    capability_ids=(),
    frames=8,
    image_size=96,
    clip_seconds=2.0,
    max_clips_per_video=1,
    max_clips_per_class_split=50,
):
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    configuration = {"source_root": str(source_root), "classes": list(classes), "domain": domain,
                     "license": license_text, "capability_ids": list(capability_ids), "frames": frames,
                     "image_size": image_size, "clip_seconds": clip_seconds,
                     "max_clips_per_video": max_clips_per_video,
                     "max_clips_per_class_split": max_clips_per_class_split}
    if (output / "manifest.json").exists():
        existing, _ = validate_manifest(output / "manifest.json")
        if existing.get("provenance", {}).get("preparation") != configuration:
            raise ValueError("Existing dataset preparation differs; choose a new version directory")
        return output / "manifest.json"
    if shutil.disk_usage(output.parent).free < 512 * 1024**2:
        raise ValueError("At least 512 MiB free space is required for bounded scene import")
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError("Provide at least two distinct classes")
    contract = {"frames": frames, "entities": 0, "image_size": image_size, "clip_seconds": clip_seconds}
    clips = discover_clips(source_root, classes, contract, max_clips_per_video, max_clips_per_class_split)
    descriptor = {
        "classes": classes,
        "domain": domain,
        "license": license_text,
        "source_root": str(source_root),
        "input_contract": contract,
        "synthetic": False,
        "source_independence_verified": False,
        "label_scope": "inherited_video_label",
        "provenance": {"preparation": configuration, "temporal_annotations_reviewed": False},
        "capability_ids": list(capability_ids),
        "clips": clips,
    }
    descriptor_path = output.parent / f"{output.name}-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    return import_scene(descriptor_path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root")
    parser.add_argument("output")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--license", required=True, dest="license_text")
    parser.add_argument("--capability-id", action="append", default=[])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--max-clips-per-video", type=int, default=1)
    parser.add_argument("--max-clips-per-class-split", type=int, default=50)
    args = parser.parse_args()
    print(
        prepare(
            args.source_root,
            args.output,
            args.classes,
            args.domain,
            args.license_text,
            args.capability_id,
            args.frames,
            args.image_size,
            args.clip_seconds,
            args.max_clips_per_video,
            args.max_clips_per_class_split,
        )
    )


if __name__ == "__main__":
    main()
