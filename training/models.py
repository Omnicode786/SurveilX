"""Custom entity-centric research model; no pretrained surveillance accuracy implied."""

import torch
from torch import nn
from torch.nn import functional as F


class ContextAdapter(nn.Module):
    def __init__(self, context_dim, width):
        super().__init__()
        self.affine = nn.Linear(context_dim, 2 * width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, entities, context):
        scale, shift = self.affine(context).chunk(2, dim=-1)
        return entities * (1 + scale[:, None]) + shift[:, None]


class SVANet(nn.Module):
    def __init__(
        self,
        width=32,
        classes=2,
        context_dim=4,
        use_motion=True,
        use_context=True,
        use_interactions=True,
        use_temporal=True,
    ):
        super().__init__()
        self.use_motion, self.use_context = use_motion, use_context
        self.use_interactions, self.use_temporal = use_interactions, use_temporal
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.SiLU(),
        )
        self.motion = nn.Linear(width, width)
        self.trajectory = nn.Linear(4, width)
        self.temporal = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.context = ContextAdapter(context_dim, width)
        self.relations = nn.Sequential(nn.Linear(2 * width + 4, width), nn.SiLU(), nn.Linear(width, width))
        self.state_head = nn.Linear(width, classes)
        self.relation_head = nn.Linear(width, 2)
        self.event_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, classes))

    def forward(self, clips, boxes, context):
        # clips: B,T,3,H,W; boxes: B,T,N,4 normalized xyxy; context: B,4.
        b, t, channels, h, w = clips.shape
        n = boxes.shape[2]
        features = self.encoder(clips.reshape(b * t, channels, h, w))
        centers = (boxes[..., :2] + boxes[..., 2:]) / 2
        grid = centers.reshape(b * t, n, 1, 2) * 2 - 1
        entities = F.grid_sample(features, grid, align_corners=False).squeeze(-1)
        entities = entities.transpose(1, 2).reshape(b, t, n, -1)
        if self.use_motion:
            delta = torch.cat([torch.zeros_like(entities[:, :1]), entities[:, 1:] - entities[:, :-1]], 1)
            velocity = torch.cat([torch.zeros_like(boxes[:, :1]), boxes[:, 1:] - boxes[:, :-1]], 1)
            entities = entities + self.motion(delta) + self.trajectory(velocity)
        if self.use_temporal:
            sequence = entities.permute(0, 2, 3, 1).reshape(b * n, -1, t)
            entities = self.temporal(sequence).reshape(b, n, -1, t).permute(0, 3, 1, 2)
        summary = entities.mean(1)
        if self.use_context:
            summary = self.context(summary, context)
        left = summary[:, :, None].expand(-1, -1, n, -1)
        right = summary[:, None, :].expand(-1, n, -1, -1)
        geometry = boxes[:, -1, :, None] - boxes[:, -1, None, :]
        relations = self.relations(torch.cat([left, right, geometry], -1))
        if self.use_interactions:
            summary = summary + relations.mean(2)
        return {
            "event": self.event_head(summary.mean(1)),
            "states": self.state_head(summary),
            "relations": self.relation_head(relations),
            "embeddings": summary,
        }


class RiskConditioner(nn.Module):
    """Experimental YOLO feature adapter. Integration/training against a detector is separate."""

    def __init__(self, channels, context_dim=4):
        super().__init__()
        self.condition = nn.Linear(context_dim, channels * 2)
        self.zone_gate = nn.Conv2d(1, channels, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        nn.init.zeros_(self.zone_gate.weight)
        nn.init.zeros_(self.zone_gate.bias)

    def forward(self, features, context, zone_map):
        scale, shift = self.condition(context).chunk(2, -1)
        zones = F.interpolate(zone_map, features.shape[-2:], mode="bilinear", align_corners=False)
        return (
            features * (1 + scale[:, :, None, None] + self.zone_gate(zones).tanh()) + shift[:, :, None, None]
        )
