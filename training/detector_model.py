"""SVA-Detector: independently initialized, context-aware multiscale detection.

Uses established feature pyramids, focal loss and GIoU; it is not a YOLO fork.
The architecture is a trainable candidate, not evidence of benchmark superiority.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torchvision.ops import nms as native_nms
except (ImportError, RuntimeError, OSError):
    native_nms = None


class ConvNorm(nn.Sequential):
    def __init__(self, incoming, outgoing, kernel=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.GroupNorm(math.gcd(8, outgoing), outgoing),
            nn.SiLU(inplace=True),
        )


class ResidualMixer(nn.Module):
    """Spatial depthwise mixing with expanded channel mixing and residual identity."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(
            ConvNorm(channels, channels, 5, groups=channels),
            ConvNorm(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(math.gcd(8, channels), channels),
        )

    def forward(self, features):
        return F.silu(features + self.mix(features))


class SceneZoneConditioner(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.scene = nn.Linear(context_dim, 2 * channels)
        self.zone = nn.Conv2d(1, channels, 1, bias=False)
        nn.init.zeros_(self.scene.weight)
        nn.init.zeros_(self.scene.bias)
        nn.init.zeros_(self.zone.weight)

    def forward(self, features, context, zones):
        scale, shift = self.scene(context).chunk(2, dim=-1)
        zones = F.interpolate(zones, size=features.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.zone(zones).tanh()
        return features * (1 + scale[:, :, None, None].tanh() + gate) + shift[:, :, None, None]


class SVADetector(nn.Module):
    """Scratch detector with strides 8/16/32 and reusable detection features.

    Boxes are normalized xyxy. Context is [B,context_dim], zones [B,1,H,W].
    ``width`` and ``depth`` scale capacity independently of input resolution.
    Embeddings share the supervised classification tower; no untrained identity
    projection is represented as a re-identification model.
    """

    def __init__(self, classes=1, width=24, depth=2, context_dim=4):
        super().__init__()
        if classes < 1 or width < 8 or depth < 1 or context_dim < 1:
            raise ValueError("classes/context_dim/depth must be positive and width at least 8")
        self.config = dict(classes=classes, width=width, depth=depth, context_dim=context_dim)
        self.stem = nn.Sequential(ConvNorm(3, width, stride=2), ConvNorm(width, width * 2, stride=2))
        self.stages = nn.ModuleList()
        previous = width * 2
        for channels in (width * 4, width * 6, width * 8):
            self.stages.append(
                nn.Sequential(
                    ConvNorm(previous, channels, stride=2),
                    *(ResidualMixer(channels) for _ in range(depth)),
                )
            )
            previous = channels
        feature_width = width * 4
        self.laterals = nn.ModuleList(
            ConvNorm(c, feature_width, 1) for c in (width * 4, width * 6, width * 8)
        )
        self.pyramid = nn.ModuleList(ResidualMixer(feature_width) for _ in range(3))
        self.conditioners = nn.ModuleList(SceneZoneConditioner(feature_width, context_dim) for _ in range(3))
        self.class_tower = nn.Sequential(ConvNorm(feature_width, feature_width), ResidualMixer(feature_width))
        self.box_tower = nn.Sequential(ConvNorm(feature_width, feature_width), ResidualMixer(feature_width))
        self.classifier = nn.Conv2d(feature_width, classes, 1)
        self.objectness = nn.Conv2d(feature_width, 1, 1)
        self.regressor = nn.Conv2d(feature_width, 4, 1)
        nn.init.constant_(self.objectness.bias, math.log(0.01 / 0.99))
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, images, context=None, zone_map=None):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError("images must have shape [B,3,H,W] with dimensions at least 32")
        b = images.shape[0]
        if context is None:
            context = images.new_zeros((b, self.config["context_dim"]))
        if zone_map is None:
            zone_map = images.new_zeros((b, 1, *images.shape[-2:]))
        if context.shape != (b, self.config["context_dim"]) or zone_map.shape[:2] != (b, 1):
            raise ValueError("context or zone_map has an incompatible shape")
        features, levels = self.stem(images), []
        for stage, lateral in zip(self.stages, self.laterals):
            features = stage(features)
            levels.append(lateral(features))
        for index in (1, 0):
            levels[index] = levels[index] + F.interpolate(
                levels[index + 1], levels[index].shape[-2:], mode="nearest"
            )
        output = []
        for index, features in enumerate(levels):
            features = self.conditioners[index](self.pyramid[index](features), context, zone_map)
            entity = self.class_tower(features)
            geometry = self.box_tower(features)
            output.append(
                {
                    "box_raw": self.regressor(geometry),
                    "objectness": self.objectness(geometry),
                    "classes": self.classifier(entity),
                    "embeddings": entity,
                }
            )
        return output


def flatten_predictions(levels):
    """Decode center offsets and positive sizes without severing autograd."""
    result = {
        key: [] for key in ("boxes", "objectness", "classes", "embeddings", "points", "steps", "levels")
    }
    for index, level in enumerate(levels):
        raw = level["box_raw"].flatten(2).transpose(1, 2)
        _, _, height, width = level["box_raw"].shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=raw.device, dtype=raw.dtype),
            torch.arange(width, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        points = torch.stack(((xx.flatten() + 0.5) / width, (yy.flatten() + 0.5) / height), -1)
        steps = raw.new_tensor([1 / width, 1 / height]).expand(height * width, -1)
        centers = points[None] + 2 * raw[..., :2].tanh() * steps[None]
        sizes = F.softplus(raw[..., 2:]) * steps[None] * 4
        result["boxes"].append(torch.cat((centers - sizes / 2, centers + sizes / 2), -1))
        for key in ("objectness", "classes", "embeddings"):
            result[key].append(level[key].flatten(2).transpose(1, 2))
        result["points"].append(points)
        result["steps"].append(steps)
        result["levels"].append(torch.full((height * width,), index, dtype=torch.long, device=raw.device))
    return {
        key: torch.cat(value, dim=0 if key in ("points", "steps", "levels") else 1)
        for key, value in result.items()
    }


def box_iou(left, right):
    intersection = (
        (
            torch.minimum(left[:, None, 2:], right[None, :, 2:])
            - torch.maximum(left[:, None, :2], right[None, :, :2])
        )
        .clamp(min=0)
        .prod(-1)
    )
    area_left = (left[:, 2:] - left[:, :2]).clamp(min=0).prod(-1)
    area_right = (right[:, 2:] - right[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_left[:, None] + area_right[None, :] - intersection).clamp(min=1e-8)


def generalized_iou_loss(predicted, target):
    intersection = (
        (torch.minimum(predicted[:, 2:], target[:, 2:]) - torch.maximum(predicted[:, :2], target[:, :2]))
        .clamp(min=0)
        .prod(-1)
    )
    union = (
        (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(-1)
        + (target[:, 2:] - target[:, :2]).clamp(min=0).prod(-1)
        - intersection
    ).clamp(min=1e-8)
    enclosure = (
        (torch.maximum(predicted[:, 2:], target[:, 2:]) - torch.minimum(predicted[:, :2], target[:, :2]))
        .clamp(min=0)
        .prod(-1)
        .clamp(min=1e-8)
    )
    return 1 - intersection / union + (enclosure - union) / enclosure


def focal_loss(logits, targets):
    probabilities = logits.sigmoid()
    correct = probabilities * targets + (1 - probabilities) * (1 - targets)
    weights = 0.25 * targets + 0.75 * (1 - targets)
    return (
        weights * (1 - correct).pow(2) * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    )


def assign_targets(flat, targets):
    """Scale-aware center assignment with a nearest-center fallback for tiny boxes."""
    batch, locations, classes = flat["classes"].shape
    objectness = flat["classes"].new_zeros((batch, locations))
    class_targets = flat["classes"].new_zeros((batch, locations, classes))
    box_targets = flat["boxes"].new_zeros((batch, locations, 4))
    points, steps, levels = flat["points"], flat["steps"], flat["levels"]
    for image_index, target in enumerate(targets):
        boxes = target["boxes"].to(points)
        labels = target["labels"].to(device=points.device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        sizes = boxes[:, 2:] - boxes[:, :2]
        # Assign each object to the pyramid whose nominal box is nearest in scale.
        desired_level = (
            torch.floor(torch.log2(sizes.max(-1).values.clamp(min=1e-6) / 0.16)).long().clamp(0, 2)
        )
        distance = ((points[:, None] - centers[None]) / steps[:, None]).abs().amax(-1)
        eligible = (distance <= 1.5) & (levels[:, None] == desired_level[None])
        for box_index in range(len(boxes)):
            if not eligible[:, box_index].any():
                candidates = distance[:, box_index].masked_fill(
                    levels != desired_level[box_index], float("inf")
                )
                eligible[candidates.argmin(), box_index] = True
        # Dense overlaps prefer the smaller object, while the center fallback covers small targets.
        cost = sizes.prod(-1)[None].expand(locations, -1).masked_fill(~eligible, float("inf"))
        best_cost, matched = cost.min(-1)
        positive = best_cost.isfinite()
        selected = matched[positive]
        objectness[image_index, positive] = 1
        class_targets[image_index, positive, labels[selected]] = 1
        box_targets[image_index, positive] = boxes[selected]
    return objectness, class_targets, box_targets


def classification_loss(logits, targets, gamma=0.0, alpha=0.25):
    if not math.isfinite(gamma) or gamma < 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Focal gamma must be nonnegative and alpha strictly between zero and one")
    terms = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma == 0:
        return terms.mean()
    probability = logits.sigmoid()
    correct_probability = probability * targets + (1 - probability) * (1 - targets)
    balance = alpha * targets + (1 - alpha) * (1 - targets)
    return (balance * (1 - correct_probability).pow(gamma) * terms).mean()


def detection_loss(
    levels,
    targets,
    classification_weight=1.0,
    classification_focal_gamma=0.0,
    classification_focal_alpha=0.25,
):
    flat = flatten_predictions(levels)
    object_target, class_target, box_target = assign_targets(flat, targets)
    positive = object_target.bool()
    denominator = positive.sum().clamp(min=1)
    object_loss = focal_loss(flat["objectness"].squeeze(-1), object_target).sum() / denominator
    if positive.any():
        class_loss = classification_loss(
            flat["classes"][positive],
            class_target[positive],
            classification_focal_gamma,
            classification_focal_alpha,
        )
        box_loss = generalized_iou_loss(flat["boxes"][positive], box_target[positive]).mean()
        size_loss = F.smooth_l1_loss(flat["boxes"][positive], box_target[positive])
    else:
        class_loss = flat["classes"].sum() * 0
        box_loss = size_loss = flat["boxes"].sum() * 0
    total = object_loss + classification_weight * class_loss + 3 * box_loss + 2 * size_loss
    return {
        "loss": total,
        "objectness": object_loss,
        "classification": class_loss,
        "box_giou": box_loss,
        "box_l1": size_loss,
        "positives": denominator.detach(),
    }


def nms(boxes, scores, threshold=0.5, limit=100):
    if native_nms is not None:
        try:
            return native_nms(boxes, scores, threshold)[:limit]
        except (NotImplementedError, RuntimeError):
            # Some TorchVision builds lack kernels for the selected accelerator.
            pass
    return python_nms(boxes, scores, threshold, limit)


def python_nms(boxes, scores, threshold=0.5, limit=100):
    order, keep = scores.argsort(descending=True), []
    while order.numel() and len(keep) < limit:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[index][None], boxes[remaining])[0] <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def decode_detections(levels, score_threshold=0.05, iou_threshold=0.5, max_detections=100, pre_nms_topk=600):
    flat, decoded = flatten_predictions(levels), []
    probabilities = flat["classes"].sigmoid() * flat["objectness"].sigmoid()
    scores, labels = probabilities.max(-1)
    for index in range(len(scores)):
        selected = torch.where(scores[index] >= score_threshold)[0]
        selected = selected[scores[index, selected].argsort(descending=True)[:pre_nms_topk]]
        boxes = flat["boxes"][index, selected].clamp(0, 1)
        candidate_scores, candidate_labels = scores[index, selected], labels[index, selected]
        valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
        boxes, candidate_scores, candidate_labels, selected = (
            boxes[valid],
            candidate_scores[valid],
            candidate_labels[valid],
            selected[valid],
        )
        keep = []
        for label in candidate_labels.unique():
            members = torch.where(candidate_labels == label)[0]
            keep.extend(
                members[
                    nms(boxes[members], candidate_scores[members], iou_threshold, max_detections)
                ].tolist()
            )
        kept = torch.tensor(keep, device=boxes.device, dtype=torch.long)
        kept = kept[candidate_scores[kept].argsort(descending=True)[:max_detections]]
        decoded.append(
            {
                "boxes": boxes[kept],
                "scores": candidate_scores[kept],
                "labels": candidate_labels[kept],
                "embeddings": F.normalize(flat["embeddings"][index, selected[kept]], dim=-1),
            }
        )
    return decoded
