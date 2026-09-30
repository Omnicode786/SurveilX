"""Explicit experimental modification of an inherited Ultralytics YOLO backbone.

Not an independent detector or a novelty claim. The base pretrained detector and
its license remain part of this model. Neutral context is used when no annotated
scene/zone data is available; that experiment cannot prove risk conditioning.
"""

import torch
from torch import nn


class SceneZoneAdapter(nn.Module):
    def __init__(self, channels, context_dim=4):
        super().__init__()
        hidden = max(8, channels // 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))
        self.context = nn.Linear(context_dim, 2 * channels)
        # Identity initialization preserves the supplied detector before training.
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        self.register_buffer("scene", torch.zeros(1, context_dim), persistent=True)
        self.register_buffer("zone", torch.zeros(1, 1, 16, 16), persistent=True)

    def set_context(self, scene, zone):
        self.scene.copy_(scene.to(self.scene).reshape_as(self.scene))
        self.zone.copy_(
            torch.nn.functional.interpolate(
                zone.to(self.zone), (16, 16), mode="bilinear", align_corners=False
            )
        )

    def forward(self, features):
        context = self.scene.expand(features.shape[0], -1).to(features)
        scale, shift = self.context(context).chunk(2, dim=-1)
        zone = torch.nn.functional.interpolate(
            self.zone.to(features), features.shape[-2:], mode="bilinear", align_corners=False
        )
        zone = zone.expand(features.shape[0], -1, -1, -1)
        spatial = self.spatial(torch.cat([features.mean(1, keepdim=True), zone], 1)).tanh()
        channel = self.channel(features).tanh()
        return (
            features * (1 + 0.1 * channel + 0.1 * spatial + 0.1 * scale[:, :, None, None].tanh())
            + 0.1 * shift[:, :, None, None]
        )


class ConditionedBlock(nn.Module):
    def __init__(self, original, channels):
        super().__init__()
        self.base = original
        self.adapter = SceneZoneAdapter(channels)
        for key in ("i", "f", "type", "np"):
            if hasattr(original, key):
                setattr(self, key, getattr(original, key))

    def forward(self, features):
        return self.adapter(self.base(features))


def install_adapters(model):
    adapted = []
    for index in (4, 6, 8):
        block = model.model[index]
        if not hasattr(block, "cv2") or not hasattr(block.cv2, "conv"):
            raise ValueError(
                f"Unsupported YOLO backbone block at {index}; architecture needs explicit integration"
            )
        channels = block.cv2.conv.out_channels
        model.model[index] = ConditionedBlock(block, channels)
        adapted.append(index)
    model.surveilx_adapted_blocks = adapted
    return model


def set_model_context(model, scene, zone):
    for block in model.modules():
        if isinstance(block, SceneZoneAdapter):
            block.set_context(scene, zone)
