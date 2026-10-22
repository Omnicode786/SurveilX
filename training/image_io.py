"""Checked image decoding; optional native JPEG reduction for training views."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_detection_image(path, image_size, mode="full-v1"):
    if mode not in {"full-v1", "bounded-jpeg-v1"} or image_size < 1:
        raise ValueError("Invalid image decode contract")
    path = Path(path)
    with Image.open(path) as header:
        width, height = header.size
        jpeg = header.format == "JPEG"
    factor = 1
    if mode == "bounded-jpeg-v1" and jpeg:
        # Retain at least twice the training canvas along BOTH dimensions.
        # Elongated images are not reduced if that would discard their short axis.
        for candidate in (2, 4, 8):
            if min(width, height) // candidate >= image_size * 2:
                factor = candidate
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
    image = cv2.imread(str(path), flag)
    fallback = image is None
    if fallback:
        # Retry the SAME bytes once (e.g. transient file API failure). Never
        # silently drop a label/image or substitute a different source.
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flag)
    if image is None or not image.size:
        raise ValueError(f"Cannot decode image: {path}; mode={mode}, source={width}x{height}")
    return image, {"mode": mode, "factor": factor, "fallback": fallback,
        "source_pixels": width * height, "decoded_bytes": image.nbytes,
        "source_bgr_bytes": width * height * 3}
