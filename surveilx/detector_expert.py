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
        from training.detection_pipeline import calibrate_scores
        from surveilx.model_profiles import runtime_profile

        profile = runtime_profile(self.manifest, resolution)
        size = profile["image_size"] if profile else self.manifest["config"]["image_size"]
        calibration = profile["calibration"] if profile else self.calibration
        square = cv2.resize(frame, (size, size))
        result = self.model.predict(
            square, imgsz=size, device=self.device, conf=0.001, iou=0.5, max_det=100, verbose=False
        )[0]
        scores = calibrate_scores(result.boxes.conf.cpu().numpy(), calibration)
        threshold = calibration.get("threshold", 0.25)
        return [
            Detection(box.tolist(), self.classes[int(label)], float(score))
            for box, label, score in zip(result.boxes.xyxyn.cpu(), result.boxes.cls.cpu(), scores)
            if score >= threshold
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
        from surveilx.model_profiles import runtime_profile

        # An arbitrary size remains forbidden. Optimized artifacts may expose only
        # explicitly recalibrated, validation-selected profiles.
        profile = runtime_profile(self.manifest, resolution)
        size = profile["image_size"] if profile else self.manifest["config"]["image_size"]
        calibration = profile["calibration"] if profile else self.calibration
        image = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None].to(self.device) / 255
        scene = torch.tensor(
            context if context is not None else [0, 0, 0, 0], dtype=torch.float32, device=self.device
        )[None]
        zones = None
        if zone_map is not None:
            zone = cv2.resize(np.asarray(zone_map, dtype=np.float32), (size, size))
            zones = torch.from_numpy(zone)[None, None].to(self.device)
        with torch.inference_mode():
            decoded = decode_detections(self.model(tensor, scene, zones), score_threshold=0.001)[0]
        scores = calibrate_scores(decoded["scores"].cpu().numpy(), calibration)
        threshold = calibration.get("threshold", 0.25)
        return [
            Detection(box.tolist(), self.classes[int(label)], float(score))
            for box, label, score in zip(decoded["boxes"].cpu(), decoded["labels"].cpu(), scores)
            if score >= threshold
        ]
