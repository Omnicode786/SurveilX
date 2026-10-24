"""Per-camera selection affects observations, not model training or approvals."""
import re
import math
from pydantic import BaseModel, Field, field_validator

ALIASES = {"people": "person", "pedestrian": "person", "persons": "person",
           "fight": "fighting", "falling": "fall", "falls": "fall"}
RULE_KINDS = {"object_presence", "configured_proximity", "calibrated_speed", "wrong_way", "signal_stop_line", "possible_missing_helmet", "possible_missing_vest", "blocked_exit", "machine_state", "scene_event"}
BACKGROUND = {"normal", "background", "no_event", "non_fighting", "not_fighting", "non_fight"}
COMMON = ["person", "fire", "smoke", "firearm", "knife", "machete", "bat", "helmet",
          "safety_vest", "gloves", "mask", "fighting", "fall", "theft", "traffic_collision", "industrial_hazard"]


def label_key(label):
    value = re.sub(r"[\s-]+", "_", str(label).strip().lower())
    return ALIASES.get(value, value)


def category(label):
    label = label_key(label)
    if label in RULE_KINDS:
        return "Alert rules"
    if label == "person":
        return "People"
    if label in {"fire", "smoke"}:
        return "Fire and smoke"
    if any(word in label for word in ("gun", "firearm", "pistol", "rifle", "knife", "machete", "bat", "weapon")):
        return "Weapons"
    if set(label.split("_")).intersection({"helmet", "vest", "glove", "gloves", "mask", "goggle", "goggles", "ear", "boot", "boots", "shoe", "shoes", "suit", "head", "hand", "hands", "face", "foot", "body", "protect", "safety", "glasses"}):
        return "Protective equipment and body parts"
    if label in {"fighting", "fall", "theft", "traffic_collision", "industrial_hazard"}:
        return "Video events"
    if label in {"car", "truck", "bus", "bicycle", "motorcycle", "train", "boat", "airplane", "traffic_light", "stop_sign", "parking_meter"}:
        return "Vehicles and traffic"
    if label in {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}:
        return "Animals"
    return "Other objects"


class DetectionSelection(BaseModel):
    disabled_labels: list[str] = Field(default_factory=list, max_length=256)

    @field_validator("disabled_labels")
    @classmethod
    def normalize_labels(cls, values):
        normalized = sorted({label_key(value) for value in values})
        if any(not re.fullmatch(r"[a-z0-9_]{1,80}", value) or value in BACKGROUND for value in normalized):
            raise ValueError("Choose valid detection labels; background classes cannot be disabled")
        return normalized

    def enabled(self, label):
        return label_key(label) not in self.disabled_labels

    def detections(self, values):
        return [value for value in values if self.enabled(value.label)]

    def event(self, evidence):
        if not evidence or not evidence.get("classes") or not self.disabled_labels:
            return evidence
        classes, probabilities = evidence.get("classes", []), evidence.get("probabilities", [])
        if any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
            return None
        if len(classes) != len(probabilities):
            return None
        kept = [(label, score) for label, score in zip(classes, probabilities) if self.enabled(label)]
        if not any(label_key(label) not in BACKGROUND for label, _ in kept):
            return None
        # Preserve original scores. Removed probability mass is never redistributed.
        return {**evidence, "classes": [label for label, _ in kept],
                "probabilities": [score for _, score in kept],
                "excluded_probability_mass": sum(score for label, score in zip(classes, probabilities) if not self.enabled(label))}


def model_labels(model):
    manifest = getattr(model, "manifest", {})
    labels = manifest.get("classes", [])
    if not labels and hasattr(model, "routes"):
        labels = [route["category"] for route in model.routes]
    if not labels and getattr(model, "model", None) is not None:
        names = getattr(model.model, "names", {})
        labels = list(names.values()) if isinstance(names, dict) else list(names)
    if not labels and getattr(model, "name", "") == "opencv-hog-person-baseline":
        labels = ["person"]
    if not labels and getattr(model, "name", "") == "synthetic-color-fixture":
        labels = ["synthetic_entity"]
    return [label for label in labels if label_key(label) not in BACKGROUND]


def model_enabled(model, selection):
    """Unknown taxonomies stay enabled; never silently disable a legacy adapter."""
    labels = model_labels(model)
    return not labels or any(selection.enabled(label) for label in labels)


def selected_class_ids(names, selection):
    """None resets cached YOLO filters; [] means bypass inference entirely."""
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    entries = list(entries)
    kept = [int(index) for index, label in entries if selection.enabled(label)]
    return None if len(kept) == len(entries) else kept


def infer_selected(model, frame, resolution, selection):
    if not model_enabled(model, selection):
        return []
    method = getattr(model, "infer_selected", None)
    result = method(frame, resolution, selection) if callable(method) else model.infer(frame, resolution)
    return selection.detections(result)
