"""SVA v3: compact bidirectional semantics and dense quality-aware scoring.

Independent implementation of ideas from EfficientDet and GFL, not their
architectures or benchmark claims. See docs/sva-v3-design.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from training.detector_model import ConvNorm, ResidualMixer, SceneZoneConditioner


class WeightedFusion(nn.Module):
    def __init__(self, channels, inputs=2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        self.refine = ResidualMixer(channels)

    def forward(self, values):
        weights = F.softplus(self.weights)
        weights = weights / weights.sum().clamp(min=1e-6)
        return self.refine(sum(weight * value for weight, value in zip(weights, values, strict=True)))


class SVAV3Core(nn.Module):
    def __init__(self, classes, width, depth, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.stem = nn.Sequential(ConvNorm(3, width, stride=2), ConvNorm(width, width * 2, stride=2))
        self.stages = nn.ModuleList()
        previous = width * 2
        for channels in (width * 4, width * 6, width * 8):
            self.stages.append(nn.Sequential(ConvNorm(previous, channels, stride=2),
                                            *(ResidualMixer(channels) for _ in range(depth))))
            previous = channels
        channels = width * 4
        self.laterals = nn.ModuleList(ConvNorm(c, channels, 1) for c in (width * 2, width * 4, width * 6, width * 8))
        self.top_down = nn.ModuleList(WeightedFusion(channels) for _ in range(3))
        self.bottom_up = nn.ModuleList(WeightedFusion(channels, 2 if index == 2 else 3) for index in range(3))
        self.conditioners = nn.ModuleList(SceneZoneConditioner(channels, context_dim) for _ in range(4))
        self.semantic = nn.Sequential(ConvNorm(channels, channels, 1), ResidualMixer(channels))
        self.class_tower = nn.Sequential(ConvNorm(channels, channels), ResidualMixer(channels), ResidualMixer(channels))
        self.box_tower = nn.Sequential(ConvNorm(channels, channels), ResidualMixer(channels))
        self.classifier = nn.Conv2d(channels, classes, 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.regressor = nn.Conv2d(channels, 4, 1)
        self.class_dropout = nn.Dropout2d(.1)
        nn.init.constant_(self.classifier.bias, math.log(.01 / .99))
        nn.init.constant_(self.objectness.bias, math.log(.01 / .99))
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, images, context, zones):
        if context is None:
            context = images.new_zeros((len(images), self.context_dim))
        if zones is None:
            zones = images.new_zeros((len(images), 1, *images.shape[-2:]))
        if context.shape != (len(images), self.context_dim) or zones.shape[:2] != (len(images), 1):
            raise ValueError("Invalid v3 context/zone shape")
        features = self.stem(images)
        raw = [self.laterals[0](features)]
        for stage, lateral in zip(self.stages, self.laterals[1:], strict=True):
            features = stage(features)
            raw.append(lateral(features))
        top = list(raw)
        for index in (2, 1, 0):
            top[index] = self.top_down[index]([raw[index], F.interpolate(top[index + 1], raw[index].shape[-2:], mode="nearest")])
        levels = [top[0]]
        for index in range(1, 4):
            finer = F.adaptive_avg_pool2d(levels[-1], raw[index].shape[-2:])
            values = [raw[index], finer] if index == 3 else [raw[index], top[index], finer]
            levels.append(self.bottom_up[index - 1](values))
        output = []
        for index, features in enumerate(levels):
            shared = self.semantic(self.conditioners[index](features, context, zones))
            entity = self.class_tower(shared)
            geometry = self.box_tower(shared)
            output.append({"box_raw": self.regressor(geometry), "objectness": self.objectness(geometry),
                           "classes": self.classifier(self.class_dropout(entity)), "embeddings": entity,
                           "quality_scores": True})
        return output


def quality_focal_loss(logits, targets, denominator):
    """Continuous-label BCE modulated by score error, normalized by positives.

    Every class/location, including background, participates. Targets detached
    from regression prevent the classifier from manipulating IoU through loss.
    """
    error = (targets.detach() - logits.sigmoid()).abs().square()
    return (F.binary_cross_entropy_with_logits(logits, targets.detach(), reduction="none") * error).sum() / denominator


def v3_detection_loss(levels, targets, quality_warmup=0.0):
    from training.detector_model import flatten_predictions, assign_targets, generalized_iou_loss, focal_loss
    if not math.isfinite(quality_warmup) or not 0 <= quality_warmup <= 1:
        raise ValueError("Quality bootstrap must be between zero and one")
    flat = flatten_predictions(levels)
    obj, classes, boxes, owners = assign_targets(flat, targets, True, "coverage-v3")
    positive = obj.bool()
    denominator = positive.sum().clamp(min=1)
    object_loss = focal_loss(flat["objectness"].squeeze(-1), obj).sum() / denominator
    probabilities = flat["classes"].detach().sigmoid()
    telemetry = {"background_max_score_sum": probabilities.amax(-1)[~positive].sum(),
        "background_locations": (~positive).sum()}
    if positive.any():
        predicted, truth = flat["boxes"][positive], boxes[positive]
        intersection = (torch.minimum(predicted[:, 2:], truth[:, 2:]) - torch.maximum(predicted[:, :2], truth[:, :2])).clamp(min=0).prod(-1)
        union = ((predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(-1) + (truth[:, 2:] - truth[:, :2]).prod(-1) - intersection).clamp(min=1e-8)
        quality = (intersection / union).detach().clamp(0, 1)
        # Observe semantic scoring separately from localization and bootstrap targets.
        true_labels = classes[positive].argmax(-1)
        telemetry.update({"positive_iou_sum": quality.sum(),
            "positive_iou_under_half": (quality < 0.5).sum(),
            "positive_true_score_sum": probabilities[positive].gather(1, true_labels[:, None]).sum(),
            "per_class_positive_counts": torch.bincount(true_labels, minlength=classes.shape[-1])})
        quality = quality_warmup + (1 - quality_warmup) * quality
        classes[positive] *= quality[:, None]
        box_loss = generalized_iou_loss(predicted, truth).mean()
        size_loss = F.smooth_l1_loss(predicted, truth)
    else:
        box_loss = size_loss = flat["boxes"].sum() * 0
    class_loss = quality_focal_loss(flat["classes"], classes, denominator)
    assigned = sum(int(row[row >= 0].unique().numel()) for row in owners)
    return {"loss": .25 * object_loss + class_loss + 3 * box_loss + 2 * size_loss,
            "objectness": object_loss, "classification": class_loss, "box_giou": box_loss, "box_l1": size_loss,
            "positives": positive.sum().detach(),
            "positives_per_level": torch.bincount(flat["levels"].expand(len(targets), -1)[positive], minlength=4).detach(),
            "assigned_objects": assigned, "unassigned_objects": sum(len(t["labels"]) for t in targets) - assigned,
            **telemetry}
