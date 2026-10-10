"""Versioned detection geometry shared by dataset loading and runtime inference."""

import cv2
import numpy as np


def prepare_image(image, size, mode="stretch"):
    if mode not in {"stretch", "letterbox"}:
        raise ValueError("Preprocessing must be stretch or letterbox")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or size < 1:
        raise ValueError("Expected a nonempty BGR image and positive canvas size")
    height, width = image.shape[:2]
    if mode == "stretch":
        resized_width = resized_height = size
    else:
        scale = size / max(height, width)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=image.dtype)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    geometry = {"mode": mode, "size": size, "resized_width": resized_width,
                "resized_height": resized_height, "left": left, "top": top}
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), geometry


def transform_boxes(boxes, geometry, inverse=False):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if geometry["mode"] == "stretch":
        return boxes
    size = geometry["size"]
    scale = np.array([geometry["resized_width"], geometry["resized_height"]] * 2, dtype=np.float32) / size
    offset = np.array([geometry["left"], geometry["top"]] * 2, dtype=np.float32) / size
    return np.clip((boxes - offset) / scale if inverse else boxes * scale + offset, 0, 1)


def prepare_zone(zone, geometry):
    size = geometry["size"]
    if zone is None:
        return np.zeros((size, size), dtype=np.float32)
    if zone.ndim != 2 or min(zone.shape) < 1:
        raise ValueError("Zone map must be a nonempty two-dimensional image")
    resized = cv2.resize(zone, (geometry["resized_width"], geometry["resized_height"]))
    canvas = np.zeros((size, size), dtype=zone.dtype)
    left, top = geometry["left"], geometry["top"]
    canvas[top:top + geometry["resized_height"], left:left + geometry["resized_width"]] = resized
    return canvas
