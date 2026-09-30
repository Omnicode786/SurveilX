"""Reviewed evidence -> reproducible datasets. Predictions never become ground truth implicitly."""

import hashlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select

from surveilx.config import settings
from surveilx.database import Feedback, Incident, Record, audit, serialize, transaction
from surveilx.evidence import read_bundle
from training.datasets import digest, validate_manifest


class BoxLabel(BaseModel):
    box: list[float] = Field(min_length=4, max_length=4)
    label: str = Field(min_length=1, max_length=100)
    track_id: int | None = Field(default=None, ge=0)

    @field_validator("box")
    @classmethod
    def geometry(cls, values):
        if (
            not all(np.isfinite(v) and 0 <= v <= 1 for v in values)
            or values[0] >= values[2]
            or values[1] >= values[3]
        ):
            raise ValueError("Boxes need finite normalized xyxy coordinates with positive area")
        return values


class FrameLabel(BaseModel):
    index: int = Field(ge=0, le=9999)
    boxes: list[BoxLabel] = Field(default_factory=list, max_length=200)


class AnnotationInput(BaseModel):
    incident_id: str
    observation: int = Field(default=0, ge=0, le=99)
    feedback_id: str | None = None
    task: Literal["detection", "event"]
    domain: str = Field(min_length=1, max_length=100)
    source_group: str = Field(min_length=1, max_length=200)
    rights: str = Field(min_length=8, max_length=1000)
    complete_annotation: bool
    event_label: str | None = Field(default=None, max_length=100)
    context: list[float] = Field(default_factory=lambda: [0, 0, 0, 0], min_length=4, max_length=4)
    frames: list[FrameLabel] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def supervision(self):
        if not self.complete_annotation:
            raise ValueError("Confirm exhaustive annotation for the intended taxonomy before submission")
        indices = [f.index for f in self.frames]
        if indices != sorted(set(indices)):
            raise ValueError("Frame indices must be unique and increasing")
        if not all(np.isfinite(v) for v in self.context):
            raise ValueError("Context must be finite")
        if self.task == "event" and not self.event_label:
            raise ValueError("Video-event annotations require a reviewed event class")
        if self.task == "event":
            tracks = [b.track_id for b in self.frames[0].boxes]
            if not tracks or None in tracks or len(set(tracks)) != len(tracks):
                raise ValueError("Event boxes need distinct persistent track IDs")
            for frame in self.frames:
                if any(b.track_id is None for b in frame.boxes) or sorted(
                    b.track_id for b in frame.boxes
                ) != sorted(tracks):
                    raise ValueError("Every event frame must contain the same persistent tracks")
        return self


def evidence_frames(incident, observation=0, review=False):
    keys = [incident.evidence_key, *incident.details.get("additional_evidence", [])]
    if observation >= len(keys) or not keys[observation]:
        raise ValueError("Evidence observation is missing or expired")
    payload = read_bundle(keys[observation])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = "review_frames/" if review else "frames/"
        names = sorted(n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".jpg"))
        if review and not names:
            names = sorted(n for n in archive.namelist() if n.startswith("frames/") and n.endswith(".jpg"))
        frames = [archive.read(n) for n in names]
    return keys[observation], hashlib.sha256(payload).hexdigest(), names, frames


def submit_annotation(body, actor):
    with transaction() as session:
        incident = session.get(Incident, body.incident_id)
        if incident is None:
            raise ValueError("Unknown incident")
        feedback = session.get(Feedback, body.feedback_id) if body.feedback_id else None
        if body.feedback_id and (
            feedback is None or feedback.incident_id != incident.id or not feedback.reviewed
        ):
            raise ValueError("Linked feedback must be independently reviewed and belong to this incident")
        if feedback and feedback.label == "ambiguous":
            raise ValueError("Ambiguous feedback cannot become a training target")
        key, checksum, names, _ = evidence_frames(incident, body.observation)
        if body.frames[-1].index >= len(names):
            raise ValueError("Annotation refers to a frame outside this evidence bundle")
        payload = {
            **body.model_dump(),
            "author": actor,
            "state": "pending",
            "reviewer": None,
            "camera_id": incident.camera_id,
            "evidence_key": key,
            "evidence_sha256": checksum,
            "synthetic": bool(incident.details.get("synthetic", False)),
            "feedback_label": feedback.label if feedback else None,
        }
        record = Record(kind="annotation", payload=payload)
        session.add(record)
        session.flush()
        audit(session, actor, "annotation_submitted", record.id, incident_id=incident.id)
        return serialize(record)


def review_annotation(annotation_id, actor, approve, notes=""):
    with transaction() as session:
        record = session.get(Record, annotation_id)
        if not record or record.kind != "annotation":
            raise ValueError("Unknown annotation")
        if record.payload["author"] == actor:
            raise ValueError("A second person must review training annotations")
        if record.payload["state"] != "pending":
            raise ValueError("Reviewed annotations are immutable; submit a new version")
        record.payload = {
            **record.payload,
            "state": "approved" if approve else "rejected",
            "reviewer": actor,
            "reviewed_at": time.time(),
            "review_notes": notes,
        }
        audit(session, actor, "annotation_reviewed", record.id, approved=approve)
        return serialize(record)


def priority(payload):
    # A transparent review heuristic, not a probability or learned risk estimate.
    return {"missed_event": 3, "false_positive": 2, "true_event": 1}.get(payload.get("feedback_label"), 0)


class DatasetBuild(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    base_dataset: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    annotation_ids: list[str] = Field(default_factory=list, max_length=500)
    max_new: int = Field(default=100, ge=1, le=500)
    replay_limit: int = Field(default=500, ge=10, le=10000)
    seed: int = Field(default=42, ge=0)


def copy_sample(sample, root, output):
    result = dict(sample)
    for field in ("image", "file", "zone_map"):
        if not sample.get(field):
            continue
        source = (root / sample[field]).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError("Invalid replay asset path")
        # Content-addressed assets avoid replay/replay/... path growth across generations.
        target = output / "replay" / f"{digest(source)}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        result[field] = target.relative_to(output).as_posix()
        if field in {"image", "file"}:
            result["sha256"] = digest(target)
    return result


def select_replay(samples, limit, seed):
    """Deterministic round-robin over label signatures; held-out samples are never sampled away."""
    rng = np.random.default_rng(seed)
    buckets = {}
    for sample in samples:
        signature = tuple(sorted(set(sample.get("labels", [sample.get("label")]))))
        buckets.setdefault(signature, []).append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for signature in sorted(buckets, key=str):
            if buckets[signature] and len(selected) < limit:
                selected.append(buckets[signature].pop())
    return selected


def build_dataset(body, actor):
    root = (settings.data_dir / "datasets" / body.base_dataset).resolve()
    base, _ = validate_manifest(root / "manifest.json")
    if base.get("representation") == "scene_clip":
        raise ValueError(
            "Scene datasets require reviewed clip intervals via the scene importer; entity-box annotations cannot substitute for clip labels"
        )
    task = base.get("task", "event")
    destination = (settings.data_dir / "datasets" / body.name).resolve()
    if destination.exists():
        raise ValueError("Dataset version already exists")
    with transaction() as session:
        records = list(session.scalars(select(Record).where(Record.kind == "annotation")))
    requested = set(body.annotation_ids)
    used = {s.get("annotation_id") for s in base["samples"]} | set(
        base.get("adaptation", {}).get("consumed_annotation_ids", [])
    )
    selected = [
        r
        for r in records
        if r.payload.get("state") == "approved"
        and r.id not in used
        and (not requested or r.id in requested)
        and r.payload["task"] == task
        and r.payload["domain"] == base["domain"]
        and r.payload["synthetic"] == base.get("synthetic", False)
    ]
    if requested and requested != {r.id for r in selected}:
        raise ValueError(
            "Requested annotations must be approved and match the dataset task, domain and synthetic scope"
        )
    selected.sort(key=lambda r: (-priority(r.payload), r.timestamp, r.id))
    selected = selected[: body.max_new]
    if not selected:
        raise ValueError("No compatible approved annotations are available")
    heldout = [s for s in base["samples"] if s["split"] != "train"]
    heldout_groups = {s["group"] for s in heldout}
    replay = select_replay(
        [s for s in base["samples"] if s["split"] == "train"], body.replay_limit, body.seed
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adaptation-", dir=destination.parent) as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        samples = [copy_sample(s, root, output) for s in [*replay, *heldout]]
        provenance = []
        previous_provenance = base.get("adaptation", {}).get("annotations", [])
        evidence_ids = set(base.get("adaptation", {}).get("consumed_evidence_frames", []))
        for record in selected:
            annotation = record.payload
            new_ids = [f"{annotation['evidence_sha256']}:{f['index']}" for f in annotation["frames"]]
            if evidence_ids.intersection(new_ids):
                raise ValueError("Evidence frame is already included in this training lineage")
            evidence_ids.update(new_ids)
            if annotation["source_group"] in heldout_groups:
                raise ValueError("Reviewed evidence group overlaps an existing held-out split")
            payload = read_bundle(annotation["evidence_key"])
            if hashlib.sha256(payload).hexdigest() != annotation["evidence_sha256"]:
                raise ValueError("Evidence changed after annotation review")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = sorted(
                    n for n in archive.namelist() if n.startswith("frames/") and n.endswith(".jpg")
                )
                decoded = []
                for item in annotation["frames"]:
                    frame = cv2.imdecode(
                        np.frombuffer(archive.read(names[item["index"]]), np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        raise ValueError("Reviewed evidence frame cannot be decoded")
                    decoded.append(frame)
            common = {
                "split": "train",
                "group": annotation["source_group"],
                "annotation_id": record.id,
                "context": annotation["context"],
            }
            if task == "detection":
                for frame, labels in zip(decoded, annotation["frames"], strict=True):
                    target = output / f"reviewed-{record.id}-{labels['index']}.png"
                    if not cv2.imwrite(str(target), frame):
                        raise ValueError("Could not save reviewed frame")
                    categories = [box["label"] for box in labels["boxes"]]
                    if any(label not in base["classes"] for label in categories):
                        raise ValueError("Annotation label is outside the base taxonomy")
                    samples.append(
                        {
                            **common,
                            "image": target.name,
                            "sha256": digest(target),
                            "boxes": [box["box"] for box in labels["boxes"]],
                            "labels": [base["classes"].index(label) for label in categories],
                        }
                    )
            else:
                contract = base.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
                if len(decoded) != contract["frames"] or any(
                    len(f["boxes"]) != contract["entities"] for f in annotation["frames"]
                ):
                    raise ValueError("Reviewed clip does not match the base event input contract")
                if annotation["event_label"] not in base["classes"]:
                    raise ValueError("Event label is outside the base taxonomy")
                size = contract["image_size"]
                clip = (
                    np.stack(
                        [
                            cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB).transpose(
                                2, 0, 1
                            )
                            for frame in decoded
                        ]
                    ).astype(np.float32)
                    / 255
                )
                boxes = np.array(
                    [
                        [b["box"] for b in sorted(f["boxes"], key=lambda b: b["track_id"])]
                        for f in annotation["frames"]
                    ],
                    dtype=np.float32,
                )
                target = output / f"reviewed-{record.id}.npz"
                np.savez_compressed(
                    target, clip=clip, boxes=boxes, context=np.array(annotation["context"], dtype=np.float32)
                )
                samples.append(
                    {
                        **common,
                        "file": target.name,
                        "label": base["classes"].index(annotation["event_label"]),
                        "sha256": digest(target),
                    }
                )
            provenance.append(
                {
                    "annotation_id": record.id,
                    "author": annotation["author"],
                    "reviewer": annotation["reviewer"],
                    "evidence_sha256": annotation["evidence_sha256"],
                    "rights": annotation["rights"],
                    "mining_priority": priority(annotation),
                }
            )
        manifest = {
            **base,
            "name": body.name,
            "samples": samples,
            "adaptation": {
                "base_dataset": body.base_dataset,
                "base_sha256": digest(root / "manifest.json"),
                "seed": body.seed,
                "replay_count": len(replay),
                "annotations": [*previous_provenance, *provenance],
                "consumed_evidence_frames": sorted(evidence_ids),
                "consumed_annotation_ids": sorted((used - {None}) | {r.id for r in selected}),
                "heldout_policy": "Copied unchanged; reviewed evidence is train-only",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _, counts = validate_manifest(output / "manifest.json")
        # Publish only after complete validation, preserving an existing version on races.
        output.rename(destination)
    with transaction() as session:
        audit(
            session,
            actor,
            "adaptation_dataset_created",
            body.name,
            annotation_ids=[r.id for r in selected],
            counts=counts,
        )
    return {
        "name": body.name,
        "counts": counts,
        "annotation_ids": [r.id for r in selected],
        "replay_count": len(replay),
    }
