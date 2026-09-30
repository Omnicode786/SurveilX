"""Load only locally trained, hash-verified SVA candidates. Preserve semantic scope."""

import json
from pathlib import Path

import cv2
import numpy as np

from surveilx.accelerators import torch_device
from training.calibration import probabilities
from training.datasets import digest


class SVAExpert:
    def __init__(self, directory):
        import torch
        from training.models import SVANet, SVASceneNet

        self.torch = torch
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if digest(directory / "weights.pt") != self.manifest["weights_sha256"]:
            raise ValueError("Model weight checksum mismatch")
        self.device = torch_device()
        self.scene = self.manifest.get("representation") == "scene_clip"
        self.model = (SVASceneNet if self.scene else SVANet)(
            classes=len(self.manifest["classes"]), **self.manifest["architecture"]
        )
        self.model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.model.to(self.device).eval()
        self.version = directory.name

    def infer(self, history, synthetic, domain, timestamps=None):
        if synthetic != self.manifest["synthetic"]:
            return {"decision": "abstain", "reason": "Model/data synthetic scope mismatch"}
        if not synthetic and domain != self.manifest["domain"]:
            return {"decision": "abstain", "reason": "Unvalidated deployment domain"}
        contract = self.manifest.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
        frames, entities, size = contract["frames"], contract["entities"], contract["image_size"]
        if self.scene:
            if timestamps is None or len(history) != len(timestamps) or len(history) < frames:
                return {"decision": "abstain", "reason": "Timestamped scene observations required"}
            times = np.asarray(timestamps, dtype=float)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                return {"decision": "abstain", "reason": "Invalid scene observation timestamps"}
            targets = np.linspace(times[-1] - contract["clip_seconds"], times[-1], frames)
            indexes = np.abs(times[:, None] - targets).argmin(axis=0)
            tolerance = contract["clip_seconds"] / (frames - 1) / 2
            if len(set(indexes)) != frames or np.max(np.abs(times[indexes] - targets)) > tolerance:
                return {
                    "decision": "abstain",
                    "reason": "Scene sampling cadence differs from training contract",
                }
            selected = [history[i] for i in indexes]
        else:
            selected = history[-frames:]
        if len(history) < frames or any(len(item[1]) < entities for item in history[-frames:]):
            return {
                "decision": "abstain",
                "reason": f"{frames} observations of {entities} persistent entities required",
            }
        ids = [] if self.scene else sorted(item["track_id"] for item in selected[-1][1])[:entities]
        clips, boxes = [], []
        for frame, detections in selected:
            lookup = {item["track_id"]: item for item in detections}
            if not all(track in lookup for track in ids):
                return {"decision": "abstain", "reason": "Entity continuity lost"}
            if isinstance(frame, (bytes, bytearray, memoryview)):
                frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    return {"decision": "abstain", "reason": "Temporal frame decode failed"}
            image = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
            clips.append(image.transpose(2, 0, 1) / 255)
            boxes.append([lookup[track]["box"] for track in ids])
        torch = self.torch

        def tensor(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = (
                self.model(tensor([clips]), tensor([boxes]), tensor([[0 if self.scene else 1, 0, 0, 0]]))["event"].cpu().numpy()
            )
        return {
            "decision": "experimental_evidence",
            "task": "event",
            "domain": self.manifest["domain"],
            "classes": self.manifest["classes"],
            "probabilities": probabilities(logits, self.manifest["temperature"])[0].tolist(),
            "calibrated": True,
            "version": self.version,
            "warning": "Calibration applies to training-domain clips, not a guarantee on this camera",
        }
