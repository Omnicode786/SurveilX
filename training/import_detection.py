"""Convert COCO boxes, YOLO detection labels or Pascal VOC boxes to a versioned manifest.

The descriptor assigns source groups and four explicit splits. No random image
split is invented, because adjacent video frames would leak between evaluations.
"""

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from training.detection_pipeline import digest, resolve_asset, validate_detection_manifest


def asset(root, value, directory=False):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing or out-of-root source: {value}")
    return path


def dimensions(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode image: {path.name}")
    return image.shape[1], image.shape[0]


def read_coco(root, source, classes):
    data = json.loads(asset(root, source["annotations"]).read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data["categories"]}
    if len(categories) != len(data["categories"]):
        raise ValueError("Duplicate COCO category IDs")
    images = {item["id"]: item for item in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("Duplicate COCO image IDs")
    annotations = {key: [] for key in images}
    for item in data["annotations"]:
        if item["image_id"] not in images or item["category_id"] not in categories:
            raise ValueError("COCO annotation references an unknown image/category")
        annotations[item["image_id"]].append(item)
    image_root = asset(root, source["images"], directory=True)
    for key, item in images.items():
        filename = item["file_name"]
        path = resolve_asset(image_root, filename)
        width, height = dimensions(path)
        if (width, height) != (item["width"], item["height"]):
            raise ValueError(f"COCO dimensions disagree with decoded image: {filename}")
        boxes, labels = [], []
        if any(a.get("iscrowd", 0) for a in annotations[key]):
            # Omitting a crowd box while retaining the image creates false background targets.
            raise ValueError(
                "COCO crowd regions require ignore-region supervision; remove entire crowd images explicitly"
            )
        for entry in annotations[key]:
            x, y, w, h = entry["bbox"]
            boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
            labels.append(classes.index(categories[entry["category_id"]]))
        yield filename, path, boxes, labels


def read_yolo(root, source, classes):
    images = asset(root, source["images"], directory=True)
    labels = asset(root, source["annotations"], directory=True)
    for path in sorted(images.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        filename = path.relative_to(images).as_posix()
        dimensions(path)
        label_path = resolve_asset(labels, str(Path(filename).with_suffix(".txt")))
        boxes, categories = [], []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    "Expected YOLO detection class/cx/cy/w/h; polygon and pose rows need another task adapter"
                )
            category = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            if not 0 <= category < len(classes):
                raise ValueError("YOLO class ID is outside the descriptor taxonomy")
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            categories.append(category)
        yield filename, path, boxes, categories


def read_voc(root, source, classes):
    images = asset(root, source["images"], directory=True)
    annotations = asset(root, source["annotations"], directory=True)
    for annotation in sorted(annotations.glob("*.xml")):
        text = annotation.read_text(encoding="utf-8")
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError("XML entities/DOCTYPE are unsupported")
        record = ET.fromstring(text)
        filename = record.findtext("filename")
        path = resolve_asset(images, filename)
        width, height = dimensions(path)
        if (width, height) != (int(record.findtext("size/width")), int(record.findtext("size/height"))):
            raise ValueError("VOC dimensions disagree with image")
        boxes, labels = [], []
        for obj in record.findall("object"):
            if obj.findtext("difficult", "0") == "1":
                raise ValueError(
                    "VOC difficult objects require ignore-region support; curate those images explicitly"
                )
            box = obj.find("bndbox")
            # VOC uses one-based inclusive pixel coordinates. Convert to zero-based edges.
            boxes.append(
                [
                    (float(box.findtext("xmin")) - 1) / width,
                    (float(box.findtext("ymin")) - 1) / height,
                    float(box.findtext("xmax")) / width,
                    float(box.findtext("ymax")) / height,
                ]
            )
            labels.append(classes.index(obj.findtext("name")))
        yield filename, path, boxes, labels


READERS = {"coco": read_coco, "yolo": read_yolo, "voc": read_voc}


def convert(descriptor, destination):
    descriptor, destination = Path(descriptor).resolve(), Path(destination).resolve()
    config = json.loads(descriptor.read_text(encoding="utf-8"))
    if destination.exists():
        raise ValueError("Choose a new dataset version directory")
    if config.get("format") not in READERS or not config.get("license") or not config.get("domain"):
        raise ValueError("Descriptor requires format=coco|yolo|voc, license and domain")
    classes = config.get("classes", [])
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Supply an explicit unique, ordered class taxonomy")
    manifest = {
        "schema_version": 1,
        "task": "detection",
        "name": destination.name,
        "classes": classes,
        "domain": config["domain"],
        "capability_ids": config.get("capability_ids", []),
        "license": config["license"],
        "synthetic": False,
        "source_format": config["format"],
        "annotation_projection": "axis-aligned detection boxes only; masks/keypoints are not trained",
        "descriptor_sha256": digest(descriptor),
        "samples": [],
    }
    with tempfile.TemporaryDirectory(prefix="surveilx-convert-") as temporary:
        root = Path(temporary)
        (root / "images").mkdir()
        for source in config["sources"]:
            for filename, image, boxes, labels in READERS[config["format"]](
                descriptor.parent, source, classes
            ):
                group = source.get("groups", {}).get(filename, source.get("group"))
                if not group:
                    raise ValueError(f"Missing camera/session/source group for {filename}")
                target = root / "images" / f"{len(manifest['samples']):07d}.png"
                # Decode and canonicalize formats so all consumers see identical pixel geometry.
                if not cv2.imwrite(str(target), cv2.imread(str(image))):
                    raise ValueError("Image conversion failed")
                manifest["samples"].append(
                    {
                        "image": target.relative_to(root).as_posix(),
                        "boxes": boxes,
                        "labels": labels,
                        "split": source["split"],
                        "group": group,
                        "source_image": filename,
                        "sha256": digest(target),
                    }
                )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _, counts, _ = validate_detection_manifest(root / "manifest.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, destination)
    return {"dataset": destination.name, "counts": counts, "classes": classes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor")
    parser.add_argument("destination")
    arguments = parser.parse_args()
    print(json.dumps(convert(arguments.descriptor, arguments.destination), indent=2))
