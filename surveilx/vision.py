import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: list[float]
    label: str
    score: float
    track_id: int | None = None

    def json(self):
        return asdict(self)


class HOGDetector:
    name = "opencv-hog-person-baseline"
    calibrated = False

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def infer(self, frame, resolution=320):
        height, width = frame.shape[:2]
        scale = resolution / width
        resized = cv2.resize(frame, (resolution, max(128, int(height * scale))))
        boxes, weights = self.hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.08)
        rh, rw = resized.shape[:2]
        return [
            Detection(
                [float(x / rw), float(y / rh), float((x + w) / rw), float((y + h) / rh)],
                "person",
                float(score),
            )
            for (x, y, w, h), score in zip(boxes, weights)
        ]


class SyntheticDetector:
    """Explicit test fixture: detects colored generated entities, not people or threats."""

    name = "synthetic-color-fixture"
    calibrated = False

    def infer(self, frame, resolution=320):
        mask = cv2.inRange(frame, np.array([40, 170, 20]), np.array([120, 255, 140]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:
                results.append(
                    Detection(
                        [x / width, y / height, (x + w) / width, (y + h) / height], "synthetic_entity", 1.0
                    )
                )
        return results


class YOLODetector:
    calibrated = False

    def __init__(self, path):
        from ultralytics import YOLO
        from surveilx.accelerators import torch_device

        if not Path(path).is_file():
            raise ValueError("Provide a local reviewed detector checkpoint")
        self.model = YOLO(path)
        self.device = torch_device()
        self.name = f"yolo:{Path(path).name}"

    def infer(self, frame, resolution=320):
        result = self.model.predict(frame, imgsz=resolution, device=self.device, verbose=False)[0]
        return [
            Detection(box.tolist(), result.names[int(cls)], float(score))
            for box, cls, score in zip(
                result.boxes.xyxyn.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu()
            )
        ]


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def area(x):
        return max(0, x[2] - x[0]) * max(0, x[3] - x[1])

    return intersection / max(1e-9, area(a) + area(b) - intersection)


class Tracker:
    """Greedy class-aware IoU baseline, explicitly not ByteTrack or identity recognition."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now=None):
        now = now if now is not None else time.monotonic()
        self.tracks = {key: value for key, value in self.tracks.items() if now - value[1] < 2}
        unused = set(self.tracks)
        for detection in detections:
            choices = [
                (iou(detection.box, self.tracks[key][0].box), key)
                for key in unused
                if self.tracks[key][0].label == detection.label
            ]
            score, key = max(choices, default=(0, 0))
            if score < 0.2:
                key = self.next_id
                self.next_id += 1
            else:
                unused.remove(key)
            detection.track_id = key
            self.tracks[key] = (detection, now)
        return detections


def generated_frame(camera, tick, width=640, height=360):
    frame = np.full((height, width, 3), (26, 24, 20), dtype=np.uint8)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (45, 43, 39), 1)
    cv2.rectangle(frame, (int(width * 0.6), 80), (width - 30, height - 40), (60, 70, 100), 2)
    x = int((tick * 28 + camera * 75) % (width - 50))
    cv2.rectangle(frame, (x, 140), (x + 36, 220), (70, 220, 90), -1)
    cv2.putText(
        frame,
        f"GENERATED TEST SOURCE / {camera + 1:02d}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 185, 190),
        1,
    )
    return frame


def redact(frame, masks):
    frame = frame.copy()
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in masks:
        frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)] = 0
    return frame
