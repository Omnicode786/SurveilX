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
    def __init__(self, original, channels, adapter_version="v1"):
        super().__init__()
        self.base = original
        self.adapter = (LocalDetailAdapter if adapter_version == "local-v2" else SceneZoneAdapter)(channels)
        for key in ("i", "f", "type", "np"):
            if hasattr(original, key):
                setattr(self, key, getattr(original, key))

    def forward(self, features):
        return self.adapter(self.base(features))


class LocalDetailAdapter(SceneZoneAdapter):
    """Bounded per-channel spatial refinement; starts as the preserved v1 adapter.

    The existing spatial mask is shared across channels and its channel mask is
    globally pooled. This additional depthwise branch can represent a different
    local response in each channel without a new detector head or dense conv.
    It is an experimental capacity extension, not a demonstrated accuracy gain.
    """

    def __init__(self, channels, context_dim=4):
        super().__init__(channels, context_dim)
        self.local_detail = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.zeros_(self.local_detail.weight)

    def forward(self, features):
        return super().forward(features) + 0.1 * features * self.local_detail(features).tanh()


def adapter_version(model):
    versions = {"local-v2" if isinstance(block.adapter, LocalDetailAdapter) else "v1"
                for block in model.modules() if isinstance(block, ConditionedBlock)}
    if len(versions) != 1:
        raise ValueError("Expected one consistent modified-YOLO adapter architecture")
    return versions.pop()


def upgrade_adapters(model, version):
    if version not in {"v1", "local-v2"}:
        raise ValueError("Unknown modified-YOLO adapter architecture")
    current = adapter_version(model)
    if current == version:
        return model
    if version == "v1":
        raise ValueError("Refusing to discard trained local-detail parameters")
    for block in model.modules():
        if isinstance(block, ConditionedBlock):
            old = block.adapter
            channels = old.channel[-1].out_channels
            parameter = next(old.parameters())
            upgraded = LocalDetailAdapter(channels).to(device=parameter.device, dtype=parameter.dtype)
            incompatible = upgraded.load_state_dict(old.state_dict(), strict=False)
            if incompatible.missing_keys != ["local_detail.weight"] or incompatible.unexpected_keys:
                raise ValueError("Adapter upgrade failed to restore every existing parameter")
            upgraded.train(old.training)
            block.adapter = upgraded
    return model


def install_adapters(model, version="v1"):
    if version not in {"v1", "local-v2"}:
        raise ValueError("Unknown modified-YOLO adapter architecture")
    adapted = []
    for index in (4, 6, 8):
        block = model.model[index]
        if not hasattr(block, "cv2") or not hasattr(block.cv2, "conv"):
            raise ValueError(
                f"Unsupported YOLO backbone block at {index}; architecture needs explicit integration"
            )
        channels = block.cv2.conv.out_channels
        model.model[index] = ConditionedBlock(block, channels, version)
        adapted.append(index)
    model.surveilx_adapted_blocks = adapted
    return model


def set_model_context(model, scene, zone):
    for block in model.modules():
        if isinstance(block, SceneZoneAdapter):
            block.set_context(scene, zone)
