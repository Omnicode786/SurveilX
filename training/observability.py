"""Small, training-independent helpers for transparent detector histories."""

import math


LOSS_COMPONENTS = ("loss", "objectness", "classification", "box_giou", "box_l1")


class EpochLossSummary:
    """Record mean batch objectives and assignment counts without retaining graphs.

    Losses retain the historical arithmetic mean of batch objectives. Assignment
    counts include repeated images selected by weighted sampling.
    """

    def __init__(self):
        self.totals = dict.fromkeys(LOSS_COMPONENTS, 0.0)
        self.counts = dict.fromkeys(LOSS_COMPONENTS, 0)
        self.batches = 0
        self.images = 0
        self.positives = 0
        self.per_level = None
        self.objects = {}
        self.score_totals = {}
        self.class_positives = None
        self.decode_totals = {"images": 0, "reduced_images": 0, "fallback_images": 0,
            "source_bgr_bytes": 0, "decoded_bytes": 0, "maximum_decoded_bytes": 0}

    def record_decodes(self, targets):
        for target in targets:
            row = target.get("image_decode")
            if not row:
                continue
            self.decode_totals["images"] += 1
            self.decode_totals["reduced_images"] += row["factor"] > 1
            self.decode_totals["fallback_images"] += bool(row["fallback"])
            for key in ("source_bgr_bytes", "decoded_bytes"):
                self.decode_totals[key] += int(row[key])
            self.decode_totals["maximum_decoded_bytes"] = max(self.decode_totals["maximum_decoded_bytes"], row["decoded_bytes"])

    def update(self, result, images):
        self.batches += 1
        self.images += images
        for key in LOSS_COMPONENTS:
            if key in result:
                value = float(result[key].detach() if hasattr(result[key], "detach") else result[key])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {key} cannot be recorded as a valid epoch")
                self.totals[key] += value
                self.counts[key] += 1
        self.positives += int(result.get("positives", 0))
        if "positives_per_level" in result:
            values = [int(value) for value in result["positives_per_level"]]
            if self.per_level is None:
                self.per_level = [0] * len(values)
            if len(self.per_level) != len(values):
                raise ValueError("FPN level count changed within an epoch")
            self.per_level = [a + b for a, b in zip(self.per_level, values, strict=True)]
        for key in ("assigned_objects", "unassigned_objects"):
            if key in result:
                self.objects[key] = self.objects.get(key, 0) + int(result[key])
        for key in ("positive_iou_sum", "positive_iou_under_half", "positive_true_score_sum",
                    "background_max_score_sum", "background_locations"):
            if key in result:
                self.score_totals[key] = self.score_totals.get(key, 0) + float(result[key])
        if "per_class_positive_counts" in result:
            counts = [int(value) for value in result["per_class_positive_counts"]]
            if self.class_positives is None:
                self.class_positives = [0] * len(counts)
            if len(self.class_positives) != len(counts):
                raise ValueError("Class count changed within epoch telemetry")
            self.class_positives = [a + b for a, b in zip(self.class_positives, counts, strict=True)]

    def report(self):
        return {
            "losses": {
                key: self.totals[key] / self.counts[key] if self.counts[key] else None
                for key in LOSS_COMPONENTS
            },
            "loss_aggregation": "arithmetic_mean_of_batch_objectives",
            "batches": self.batches,
            "sampled_images": self.images,
            "positive_assignments": self.positives,
            "positives_per_level": self.per_level,
            "assignment_scope": "sum over sampled images; includes sampler repetitions",
            "image_decode": dict(self.decode_totals) if self.decode_totals["images"] else None,
            "score_quality": {
                "positive_iou_mean": self.score_totals.get("positive_iou_sum", 0) / max(1, self.positives),
                "positive_iou_under_half": self.score_totals.get("positive_iou_under_half", 0),
                "positive_true_class_score_mean": self.score_totals.get("positive_true_score_sum", 0) / max(1, self.positives),
                "background_max_class_score_mean": self.score_totals.get("background_max_score_sum", 0) / max(1, self.score_totals.get("background_locations", 0)),
                "per_class_positive_counts": self.class_positives,
                "scope": "Positive-location weighted raw v3 quality scores; not calibrated probability or AP",
            } if self.score_totals else None,
            **self.objects,
        }


class ValidationSelection:
    """Preserve AP50-first selection and explicitly track the parent at epoch zero."""

    def __init__(self, patience=None, parent_map50=None, parent_loss=None):
        if patience is not None and (isinstance(patience, bool) or patience < 1):
            raise ValueError("Patience must be positive or None")
        if (parent_map50 is None) != (parent_loss is None):
            raise ValueError("Parent AP50 and loss must be supplied together")
        self.patience = patience
        self.best = (-1.0, -float("inf"))
        self.best_epoch = self.stale_epochs = 0
        self.selected = "candidate"
        if parent_map50 is not None:
            self.best = self.score(parent_map50, parent_loss)
            self.selected = "parent"

    @staticmethod
    def score(map50, loss):
        if map50 is None or not math.isfinite(map50) or not 0 <= map50 <= 1:
            raise ValueError("Validation AP50 must be finite and between zero and one")
        if loss is None or not math.isfinite(loss):
            raise ValueError("Validation loss must be finite")
        return float(map50), -float(loss)

    def observe(self, epoch, map50, loss):
        score = self.score(map50, loss)
        improved = score > self.best
        if improved:
            self.best, self.best_epoch, self.stale_epochs = score, epoch, 0
            self.selected = "candidate"
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.patience is not None and self.stale_epochs >= self.patience

    def report(self, epoch, maximum):
        reason = (
            "validation_patience_exhausted"
            if self.should_stop
            else "maximum_epochs_reached" if epoch >= maximum else None
        )
        return {
            "criterion": "validation_AP50_then_lower_reference_BCE_loss",
            "best_epoch": self.best_epoch,
            "best_map50": self.best[0],
            "best_validation_loss": -self.best[1],
            "selected": self.selected,
            "stale_epochs": self.stale_epochs,
            "patience": self.patience,
            "stopping_reason": reason,
        }
