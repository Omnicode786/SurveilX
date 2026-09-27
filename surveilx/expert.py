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
        from training.models import SVANet

        self.torch = torch
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if digest(directory / "weights.pt") != self.manifest["weights_sha256"]:
            raise ValueError("Model weight checksum mismatch")
        self.device = torch_device()
        self.model = SVANet(classes=len(self.manifest["classes"]), **self.manifest["architecture"])
        self.model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.model.to(self.device).eval()
        self.version = directory.name

    def infer(self, history, synthetic, domain):
        if synthetic != self.manifest["synthetic"]:
            return {"decision": "abstain", "reason": "Model/data synthetic scope mismatch"}
        if not synthetic and domain != self.manifest["domain"]:
            return {"decision": "abstain", "reason": "Unvalidated deployment domain"}
        if len(history) < 8 or any(len(item[1]) < 2 for item in history[-8:]):
            return {"decision": "abstain", "reason": "Eight observations of two persistent entities required"}
        selected = history[-8:]
        ids = sorted(item["track_id"] for item in selected[-1][1])[:2]
        clips, boxes = [], []
        for frame, detections in selected:
            lookup = {item["track_id"]: item for item in detections}
            if not all(track in lookup for track in ids):
                return {"decision": "abstain", "reason": "Entity continuity lost"}
            image = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2RGB)
            clips.append(image.transpose(2, 0, 1) / 255)
            boxes.append([lookup[track]["box"] for track in ids])
        torch = self.torch

        def tensor(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = (
                self.model(tensor([clips]), tensor([boxes]), tensor([[1, 0, 0, 0]]))["event"].cpu().numpy()
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
