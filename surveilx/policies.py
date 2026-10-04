"""Configured visual observations requiring review, never intent or legal findings."""

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SiteRule(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    kind: Literal[
        "object_presence",
        "restricted_zone",
        "wrong_way",
        "possible_missing_helmet",
        "possible_missing_vest",
        "configured_proximity",
        "calibrated_speed",
        "signal_stop_line",
        "machine_state",
        "blocked_exit",
        "scene_event",
    ]
    labels: list[str] = Field(default_factory=list, max_length=80)
    secondary_labels: list[str] = Field(default_factory=list, max_length=80)
    zone: list[float] = Field(default_factory=lambda: [0, 0, 1, 1], min_length=4, max_length=4)
    signal_zone: list[float] = Field(default_factory=lambda: [0, 0, 1, 1], min_length=4, max_length=4)
    stop_line: list[float] = Field(default_factory=lambda: [0.5, 0, 0.5, 1], min_length=4, max_length=4)
    direction: list[float] = Field(default_factory=lambda: [1, 0], min_length=2, max_length=2)
    ground_plane_transform: list[float] = Field(
        default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        min_length=9,
        max_length=9,
    )
    calibration_id: str | None = Field(default=None, max_length=80)
    min_displacement: float = Field(default=0.05, ge=0.01, le=1)
    max_distance: float = Field(default=0.15, gt=0, le=2)
    speed_mps: float = Field(default=5.0, gt=0, le=120)
    confidence: float = Field(default=0.6, ge=0.1, le=1)
    duration: float = Field(default=2, ge=0.5, le=60)
    max_gap: float = Field(default=3, ge=0.2, le=10)
    min_observations: int = Field(default=3, ge=2, le=100)
    cooldown: float = Field(default=60, ge=5, le=3600)

    @model_validator(mode="after")
    def valid_geometry(self):
        if not all(
            math.isfinite(v)
            for v in self.zone + self.signal_zone + self.stop_line + self.direction + self.ground_plane_transform
        ):
            raise ValueError("Finite geometry required")
        for zone in (self.zone, self.signal_zone):
            x1, y1, x2, y2 = zone
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Zone must be normalized xyxy")
        x1, y1, x2, y2 = self.stop_line
        if not all(0 <= value <= 1 for value in self.stop_line) or math.hypot(x2 - x1, y2 - y1) < 1e-6:
            raise ValueError("Stop line must be a normalized non-zero segment")
        if self.kind == "wrong_way" and math.hypot(*self.direction) < 1e-6:
            raise ValueError("Allowed direction cannot be zero")
        if self.kind not in {"possible_missing_helmet", "possible_missing_vest"} and not self.labels:
            raise ValueError("Choose the labels to observe")
        if self.kind in {"configured_proximity", "machine_state", "signal_stop_line"} and not self.secondary_labels:
            raise ValueError(f"{self.kind} requires secondary labels")
        if self.kind == "calibrated_speed" and not self.calibration_id:
            raise ValueError("Calibrated speed requires an explicit site calibration ID")
        return self


class SitePolicy(BaseModel):
    rules: list[SiteRule] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({rule.id for rule in self.rules}) != len(self.rules):
            raise ValueError("Rule IDs must be unique")
        return self


def center(item):
    x1, y1, x2, y2 = item["box"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def inside(point, zone):
    return zone[0] <= point[0] <= zone[2] and zone[1] <= point[1] <= zone[3]


def ground_point(item):
    x1, _, x2, y2 = item["box"]
    return (x1 + x2) / 2, y2


def project_ground(point, transform):
    x, y = point
    h = transform
    denominator = h[6] * x + h[7] * y + h[8]
    if abs(denominator) < 1e-9:
        return None
    return ((h[0] * x + h[1] * y + h[2]) / denominator, (h[3] * x + h[4] * y + h[5]) / denominator)


def line_side(point, line):
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crosses_segment(previous, current, line):
    """Require crossing the configured finite segment, not its infinite extension."""
    before, after = line_side(previous, line), line_side(current, line)
    if abs(before) <= 1e-9 or before * after > 0 or abs(before - after) <= 1e-9:
        return False
    fraction = before / (before - after)
    point = [previous[i] + fraction * (current[i] - previous[i]) for i in (0, 1)]
    dx, dy = line[2] - line[0], line[3] - line[1]
    along = ((point[0] - line[0]) * dx + (point[1] - line[1]) * dy) / (dx * dx + dy * dy)
    return 0 <= fraction <= 1 and 0 <= along <= 1


class PolicyEngine:
    def __init__(self, policy):
        self.policy = SitePolicy.model_validate(policy)
        self.states = {}
        self.last_alert = {}

    def evaluate(self, outputs, now, expert_key):
        observations = []
        if not math.isfinite(now):
            return observations
        seen = set()
        for rule in self.policy.rules:
            if rule.kind == "scene_event":
                continue  # Scene rules require calibrated temporal evidence, not a same-named box.
            reliable = [d for d in outputs if d["score"] >= rule.confidence]
            for item in reliable:
                if not item.get("track_id"):
                    continue
                if rule.kind != "signal_stop_line" and not inside(center(item), rule.zone):
                    continue
                if rule.kind == "possible_missing_helmet":
                    if item["label"] != "person":
                        continue
                    heads = [
                        d
                        for d in reliable
                        if d["label"] == "head"
                        and inside(center(d), item["box"])
                        and (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) >= 0.001
                    ]
                    if len(heads) != 1:
                        continue  # Missing/small/ambiguous head: abstain instead of inferring absence.
                    head = heads[0]
                    owners = [
                        d for d in reliable if d["label"] == "person" and inside(center(head), d["box"])
                    ]
                    if len(owners) != 1 or any(
                        d["label"] == "helmet" and inside(center(d), item["box"]) for d in reliable
                    ):
                        continue
                elif rule.kind == "possible_missing_vest":
                    if item["label"] != "person":
                        continue
                    torsos = [
                        d
                        for d in reliable
                        if d["label"] == "torso"
                        and inside(center(d), item["box"])
                        and (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) >= 0.003
                    ]
                    if len(torsos) != 1:
                        continue  # No explicit, unambiguous visible torso: abstain.
                    torso = torsos[0]
                    owners = [
                        d for d in reliable if d["label"] == "person" and inside(center(torso), d["box"])
                    ]
                    if len(owners) != 1 or any(
                        d["label"] == "safety_vest" and inside(center(d), torso["box"]) for d in reliable
                    ):
                        continue
                elif rule.kind == "configured_proximity":
                    if item["label"] not in rule.labels:
                        continue
                    candidates = [
                        d
                        for d in reliable
                        if d["label"] in rule.secondary_labels
                        and d.get("track_id")
                        and d["track_id"] != item["track_id"]
                        and inside(center(d), rule.zone)
                    ]
                    if not candidates:
                        continue
                    related = min(candidates, key=lambda d: math.dist(center(item), center(d)))
                    distance = math.dist(center(item), center(related))
                    if distance > rule.max_distance:
                        continue
                elif rule.kind == "calibrated_speed":
                    if item["label"] not in rule.labels:
                        continue
                    metric_point = project_ground(ground_point(item), rule.ground_plane_transform)
                    if metric_point is None:
                        continue
                    identity = item["track_id"]
                    key = (rule.id, expert_key, identity)
                    seen.add(key)
                    previous = self.states.get(key)
                    if previous is None or now - previous["last"] > rule.max_gap:
                        self.states[key] = {
                            "since": now,
                            "last": now,
                            "count": 0,
                            "metric_point": metric_point,
                            "last_speed_mps": 0.0,
                        }
                        continue
                    elapsed = now - previous["last"]
                    if elapsed <= 0:
                        continue
                    speed = math.dist(metric_point, previous["metric_point"]) / elapsed
                    previous.update(last=now, metric_point=metric_point, last_speed_mps=speed)
                    if speed < rule.speed_mps:
                        previous.update(since=now, count=0)
                        self.states[key] = previous
                        continue
                    previous["count"] += 1
                    self.states[key] = previous
                    if now - previous["since"] < rule.duration or previous["count"] < rule.min_observations:
                        continue
                    alert_key = (rule.id, expert_key)
                    if now - self.last_alert.get(alert_key, -float("inf")) < rule.cooldown:
                        continue
                    self.last_alert[alert_key] = now
                    observations.append(
                        {
                            "rule_id": rule.id,
                            "kind": rule.kind,
                            "label": item["label"],
                            "track_id": item["track_id"],
                            "observations": previous["count"],
                            "duration_seconds": now - previous["since"],
                            "metric_speed_mps": speed,
                            "speed_threshold_mps": rule.speed_mps,
                            "calibration_id": rule.calibration_id,
                            "interpretation": "Configured speed observation from calibrated ground-plane geometry requiring human review; not legal speed enforcement",
                        }
                    )
                    continue
                elif rule.kind == "signal_stop_line":
                    if item["label"] not in rule.labels:
                        continue
                    signal = next(
                        (
                            d
                            for d in reliable
                            if d["label"] in rule.secondary_labels and inside(center(d), rule.signal_zone)
                        ),
                        None,
                    )
                    if signal is None:
                        continue
                    point = ground_point(item)
                    side = line_side(point, rule.stop_line)
                    key = (rule.id, expert_key, item["track_id"])
                    seen.add(key)
                    previous = self.states.get(key)
                    if previous is None or now - previous["last"] > rule.max_gap:
                        self.states[key] = {"since": now, "last": now, "count": 0, "side": side,
                                            "point": point, "active": False}
                        continue
                    if now <= previous["last"]:
                        continue
                    crossed = crosses_segment(previous["point"], point, rule.stop_line)
                    active = (previous.get("active") or crossed) and inside(center(item), rule.zone)
                    if not active:
                        previous.update(since=now, count=0, side=side, point=point, active=False, last=now)
                        self.states[key] = previous
                        continue
                    if not previous.get("active"):
                        previous.update(since=now, count=0)
                    previous.update(last=now, side=side, point=point, active=True)
                    previous["count"] += 1
                    self.states[key] = previous
                    if now - previous["since"] < rule.duration or previous["count"] < rule.min_observations:
                        continue
                    alert_key = (rule.id, expert_key)
                    if now - self.last_alert.get(alert_key, -float("inf")) < rule.cooldown:
                        continue
                    self.last_alert[alert_key] = now
                    observations.append(
                        {
                            "rule_id": rule.id,
                            "kind": rule.kind,
                            "label": item["label"],
                            "track_id": item["track_id"],
                            "signal_label": signal["label"],
                            "observations": previous["count"],
                            "duration_seconds": now - previous["since"],
                            "interpretation": "Configured stop-line observation while a configured signal state is visible; requires human review and is not a legal finding",
                        }
                    )
                    continue
                elif rule.kind == "machine_state":
                    if item["label"] not in rule.labels:
                        continue
                    state = next(
                        (
                            d
                            for d in reliable
                            if d["label"] in rule.secondary_labels
                            and inside(center(d), rule.zone)
                            and math.dist(center(item), center(d)) <= rule.max_distance
                        ),
                        None,
                    )
                    if state is None:
                        continue
                    related = state
                    distance = math.dist(center(item), center(related))
                elif rule.kind == "blocked_exit":
                    if item["label"] not in rule.labels:
                        continue
                elif item["label"] not in rule.labels:
                    continue
                identity = (
                    (item["track_id"], related["track_id"])
                    if rule.kind == "configured_proximity"
                    else (item["track_id"], related.get("track_id"), related["label"])
                    if rule.kind == "machine_state"
                    else item["track_id"]
                )
                key = (rule.id, expert_key, identity)
                seen.add(key)
                previous = self.states.get(key)
                point = center(item)
                if previous is not None and now <= previous["last"]:
                    continue
                if previous is None or now - previous["last"] > rule.max_gap:
                    previous = {"since": now, "last": now, "count": 0, "origin": point}
                previous["count"] += 1
                previous["last"] = now
                self.states[key] = previous
                if rule.kind == "wrong_way":
                    length = math.hypot(*rule.direction)
                    displacement = sum(
                        (point[i] - previous["origin"][i]) * rule.direction[i] / length for i in (0, 1)
                    )
                    if displacement > 0:
                        previous.update(origin=point, since=now, count=1)
                    if displacement > -rule.min_displacement:
                        continue
                if now - previous["since"] < rule.duration or previous["count"] < rule.min_observations:
                    continue
                # Cooldown is per rule/artifact, not per newly assigned track ID.
                alert_key = (rule.id, expert_key)
                if now - self.last_alert.get(alert_key, -float("inf")) < rule.cooldown:
                    continue
                self.last_alert[alert_key] = now
                observations.append(
                    {
                        "rule_id": rule.id,
                        "kind": rule.kind,
                        "label": item["label"],
                        "track_id": item["track_id"],
                        "observations": previous["count"],
                        "duration_seconds": now - previous["since"],
                        "interpretation": "Configured visual observation requiring human review; not proof of intent, equipment certification or a legal violation",
                        **(
                            {
                                "related_label": related["label"],
                                "related_track_id": related.get("track_id"),
                                "normalized_center_distance": distance,
                            }
                            if rule.kind in {"configured_proximity", "machine_state"}
                            else {}
                        ),
                    }
                )
        self.states = {
            key: value
            for key, value in self.states.items()
            if (key[1] != expert_key or key in seen) and now - value["last"] <= 10
        }
        self.last_alert = {key: value for key, value in self.last_alert.items() if now - value <= 3600}
        return observations

    def evaluate_event(self, evidence, now, expert_key):
        """Persist configured calibrated clip classifications before review alerts."""
        observations = []
        def reset():
            self.states = {key: value for key, value in self.states.items()
                           if not (key[1] == expert_key and key[2] == "scene")}

        if not evidence or evidence.get("decision") != "experimental_evidence" or not evidence.get("calibrated"):
            reset()
            return observations
        classes, probabilities = evidence.get("classes", []), evidence.get("probabilities", [])
        if (not classes or len(classes) != len(probabilities) or not math.isfinite(now)
                or not all(isinstance(label, str) and label for label in classes)
                or len(classes) != len(set(classes))
                or not all(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
                           for score in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)):
            reset()
            return observations
        scores = dict(zip(classes, probabilities, strict=True))
        for rule in self.policy.rules:
            if rule.kind != "scene_event":
                continue
            matches = [(label, scores[label]) for label in rule.labels if scores.get(label, 0) >= rule.confidence]
            key = (rule.id, expert_key, "scene")
            if not matches:
                self.states.pop(key, None)
                continue
            label, score = max(matches, key=lambda item: item[1])
            previous = self.states.get(key)
            if previous is not None and now <= previous["last"]:
                continue
            if previous is None or now - previous["last"] > rule.max_gap or previous.get("label") != label:
                previous = {"since": now, "last": now, "count": 0, "label": label}
            previous["count"] += 1
            previous["last"] = now
            self.states[key] = previous
            if now - previous["since"] < rule.duration or previous["count"] < rule.min_observations:
                continue
            alert_key = (rule.id, expert_key)
            if now - self.last_alert.get(alert_key, -float("inf")) < rule.cooldown:
                continue
            self.last_alert[alert_key] = now
            observations.append(
                {
                    "rule_id": rule.id,
                    "kind": rule.kind,
                    "label": label,
                    "score": score,
                    "track_id": None,
                    "observations": previous["count"],
                    "duration_seconds": now - previous["since"],
                    "interpretation": "Persistent calibrated clip classification requiring human review; not proof that the event occurred",
                }
            )
        self.states = {
            key: value for key, value in self.states.items() if key[2] != "scene" or now - value["last"] <= 10
        }
        return observations
