"""Runtime adapter for a locally trained and explicitly accepted scratch detector."""

import json
from pathlib import Path

import cv2
import numpy as np

from surveilx.vision import Detection


class AdaptedYOLODetector:
    def __init__(self, directory):
        from ultralytics import YOLO
        from surveilx.accelerators import torch_device
        from training.detection_pipeline import digest

        directory = Path(directory).resolve()
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("architecture") not in {"yolo_rai", "yolo_baseline", "yolo_pretrained"}:
            raise ValueError("Not a supported YOLO artifact")
        if digest(directory / "weights.pt") != self.manifest.get("weights_sha256"):
            raise ValueError("YOLO checkpoint checksum mismatch")
        self.model = YOLO(str(directory / "weights.pt"))
        self.device = torch_device()
        self.name = self.manifest["architecture"]
        self.version = directory.name
        self.calibration = self.manifest.get("calibration", {})
        self.calibrated = self.calibration.get("status") == "fitted"
        self.classes = self.manifest["classes"]
        if [self.model.names[i] for i in range(len(self.model.names))] != self.classes:
            raise ValueError("Checkpoint class order differs from the artifact taxonomy")

    def infer(self, frame, resolution=320):
        from surveilx.detection_preferences import DetectionSelection
        return self.infer_selected(frame, resolution, DetectionSelection())

    def infer_selected(self, frame, resolution, selection):
        from training.detection_pipeline import calibrate_scores
        from surveilx.model_profiles import runtime_profile
        from surveilx.detection_preferences import selected_class_ids

        classes = selected_class_ids(self.classes, selection)
        if classes == []:
            return []

        profile = runtime_profile(self.manifest, resolution) if resolution is not None else None
        size = profile["image_size"] if profile else self.manifest["config"]["image_size"]
        calibration = profile["calibration"] if profile else self.calibration
        from training.detection_geometry import prepare_image, transform_boxes
        square, geometry = prepare_image(frame, size, self.manifest.get("config", {}).get("preprocessing", "stretch"))
        square = cv2.cvtColor(square, cv2.COLOR_RGB2BGR)
        result = self.model.predict(
            square, imgsz=size, device=self.device, conf=0.001, iou=0.5, max_det=100, verbose=False, classes=classes
        )[0]
        scores = calibrate_scores(result.boxes.conf.cpu().numpy(), calibration)
        threshold = calibration.get("threshold", 0.25)
        return [
            Detection(box.tolist(), self.classes[int(label)], float(score))
            for box, label, score in zip(transform_boxes(result.boxes.xyxyn.cpu().numpy(), geometry, inverse=True), result.boxes.cls.cpu(), scores)
            if score >= threshold and box[2] > box[0] and box[3] > box[1]
        ]


class ScratchDetector:
    def __init__(self, directory):
        import torch

        from surveilx.accelerators import torch_device
        from training.detection_pipeline import digest
        from training.detector_model import SVADetector

        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("task") != "detection" or self.manifest.get("architecture") != "sva-detector":
            raise ValueError("Not an SVA-Detector image detection artifact")
        if digest(self.directory / "weights.pt") != self.manifest.get("weights_sha256"):
            raise ValueError("Detector checkpoint checksum mismatch")
        self.device = torch_device()
        self.model = SVADetector(**self.manifest["model_config"]).to(self.device)
        self.model.load_state_dict(
            torch.load(self.directory / "weights.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.name = "sva-detector"
        self.version = self.directory.name
        self.calibration = self.manifest.get("calibration", {})
        self.calibrated = self.calibration.get("status") == "fitted"
        self.classes = self.manifest["classes"]

    def infer(self, frame, resolution=320, context=None, zone_map=None):
        import torch

        from training.detection_pipeline import calibrate_scores
        from training.detector_model import decode_detections
        from training.detection_geometry import prepare_image, prepare_zone, transform_boxes
        from surveilx.model_profiles import runtime_profile

        # An arbitrary size remains forbidden. Optimized artifacts may expose only
        # explicitly recalibrated, validation-selected profiles.
        profile = runtime_profile(self.manifest, resolution) if resolution is not None else None
        size = profile["image_size"] if profile else self.manifest["config"]["image_size"]
        calibration = profile["calibration"] if profile else self.calibration
        image, geometry = prepare_image(frame, size, self.manifest["config"].get("preprocessing", "stretch"))
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None].to(self.device) / 255
        scene = torch.tensor(
            context if context is not None else [0, 0, 0, 0], dtype=torch.float32, device=self.device
        )[None]
        zones = None
        if zone_map is not None:
            zone = prepare_zone(np.asarray(zone_map, dtype=np.float32), geometry)
            zones = torch.from_numpy(zone)[None, None].to(self.device)
        with torch.inference_mode():
            decoded = decode_detections(self.model(tensor, scene, zones), score_threshold=0.001)[0]
        scores = calibrate_scores(decoded["scores"].cpu().numpy(), calibration)
        threshold = calibration.get("threshold", 0.25)
        boxes = transform_boxes(decoded["boxes"].cpu().numpy(), geometry, inverse=True)
        return [
            Detection(box.tolist(), self.classes[int(label)], float(score))
            for box, label, score in zip(boxes, decoded["labels"].cpu(), scores)
            if score >= threshold and np.all(box[2:] > box[:2])
        ]
