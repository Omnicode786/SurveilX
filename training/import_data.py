"""Convert user video clips and reviewed per-frame entity boxes to the training contract."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.datasets import digest, validate_manifest


def convert(annotation_path, output):
    annotation_path, output = Path(annotation_path).resolve(), Path(output).resolve()
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    contract = annotations.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
    size = contract["image_size"]
    if output.exists():
        raise ValueError("Use a new, nonexistent dataset version directory")
    output.mkdir(parents=True)
    samples = []
    for index, item in enumerate(annotations["samples"]):
        video = (annotation_path.parent / item["video"]).resolve()
        if not video.is_relative_to(annotation_path.parent):
            raise ValueError("Videos must be inside the annotation directory")
        capture = cv2.VideoCapture(str(video))
        try:
            clip = []
            for frame_number in item["frame_indices"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Unreadable frame {frame_number} in sample {index}")
                image = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
                clip.append(image.transpose(2, 0, 1).astype("float32") / 255)
        finally:
            capture.release()
        if len(clip) != contract["frames"]:
            raise ValueError("Frame indices must match input_contract.frames")
        file = f"sample-{index:06d}.npz"
        np.savez_compressed(
            output / file,
            clip=np.stack(clip),
            boxes=np.asarray(item["boxes"], dtype="float32"),
            context=np.asarray(item["context"], dtype="float32"),
        )
        samples.append(
            {
                "file": file,
                "label": item["label"],
                "group": item["group"],
                "split": item["split"],
                "sha256": digest(output / file),
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "event",
        "input_contract": contract,
        "name": output.name,
        "synthetic": False,
        "domain": annotations["domain"],
        "capability_ids": annotations.get("capability_ids", []),
        "classes": annotations["classes"],
        "license": annotations["license"],
        "samples": samples,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations")
    parser.add_argument("output")
    args = parser.parse_args()
    print(convert(args.annotations, args.output))
