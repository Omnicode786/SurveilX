"""Configured visual observations requiring review, never intent or legal findings."""

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SiteRule(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    kind: Literal[
        "object_presence", "restricted_zone", "wrong_way", "possible_missing_helmet", "scene_event"
    ]
    labels: list[str] = Field(default_factory=list, max_length=80)
    zone: list[float] = Field(default_factory=lambda: [0, 0, 1, 1], min_length=4, max_length=4)
    direction: list[float] = Field(default_factory=lambda: [1, 0], min_length=2, max_length=2)
    min_displacement: float = Field(default=0.05, ge=0.01, le=1)
    confidence: float = Field(default=0.6, ge=0.1, le=1)
    duration: float = Field(default=2, ge=0.5, le=60)
    max_gap: float = Field(default=3, ge=0.2, le=10)
    min_observations: int = Field(default=3, ge=2, le=100)
    cooldown: float = Field(default=60, ge=5, le=3600)

    @model_validator(mode="after")
    def valid_geometry(self):
        if not all(math.isfinite(v) for v in self.zone + self.direction):
            raise ValueError("Finite geometry required")
        x1, y1, x2, y2 = self.zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Zone must be normalized xyxy")
        if self.kind == "wrong_way" and math.hypot(*self.direction) < 1e-6:
            raise ValueError("Allowed direction cannot be zero")
        if self.kind != "possible_missing_helmet" and not self.labels:
            raise ValueError("Choose the labels to observe")
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


class PolicyEngine:
    def __init__(self, policy):
        self.policy = SitePolicy.model_validate(policy)
        self.states = {}
        self.last_alert = {}

    def evaluate(self, outputs, now, expert_key):
        observations = []
        seen = set()
        for rule in self.policy.rules:
            reliable = [d for d in outputs if d["score"] >= rule.confidence]
            for item in reliable:
                if not item.get("track_id") or not inside(center(item), rule.zone):
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
                elif item["label"] not in rule.labels:
                    continue
                key = (rule.id, expert_key, item["track_id"])
                seen.add(key)
                previous = self.states.get(key)
                point = center(item)
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
        if not evidence or evidence.get("decision") != "experimental_evidence" or not evidence.get("calibrated"):
            return observations
        classes, probabilities = evidence.get("classes", []), evidence.get("probabilities", [])
        if len(classes) != len(probabilities):
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
            if previous is None or now - previous["last"] > rule.max_gap:
                previous = {"since": now, "last": now, "count": 0}
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
