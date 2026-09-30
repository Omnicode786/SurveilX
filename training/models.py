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


class SVASceneNet(nn.Module):
    """Whole-clip classification for video labels; no entity boxes or frame labels are invented."""

    def __init__(
        self, classes=2, use_motion=True, use_context=True, use_temporal=True, use_interactions=False
    ):
        super().__init__()
        if use_interactions:
            raise ValueError("Scene clips do not supervise entity interactions")
        self.use_motion, self.use_context, self.use_temporal = use_motion, use_context, use_temporal
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.motion = nn.Linear(64, 64, bias=False)
        self.context = ContextAdapter(4, 64)
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.event_head = nn.Linear(64, classes)

    def forward(self, clip, boxes=None, context=None):
        batch, frames, channels, height, width = clip.shape
        features = self.encoder(clip.reshape(batch * frames, channels, height, width)).reshape(
            batch, frames, 64
        )
        if self.use_motion:
            difference = torch.cat(
                [torch.zeros_like(features[:, :1]), features[:, 1:] - features[:, :-1]], dim=1
            )
            features = features + self.motion(difference)
        if self.use_context and context is not None:
            features = self.context(features, context)
        if self.use_temporal:
            features, _ = self.temporal(features)
        return {"event": self.event_head(features.mean(dim=1))}


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
