import io
import json
import tempfile
import uuid
import zipfile
from pathlib import Path

import cv2

from surveilx.config import settings
from surveilx.security import cipher


def client():
    import boto3

    return boto3.client("s3", endpoint_url=settings.s3_endpoint or None)


def detection_title(metadata):
    labels = metadata.get("observed_labels") or []
    if not labels and metadata.get("policy_observation", {}).get("label"):
        labels = [metadata["policy_observation"]["label"]]
    readable = [str(label).replace("_", " ").strip() for label in labels if str(label).strip()]
    return "Detected: " + ", ".join(dict.fromkeys(readable)) if readable else "Detected sequence: review required"


def titled_frame(frame, title, timestamp):
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    band = min(height, max(36, round(height * 0.1)))
    overlay = rendered.copy()
    cv2.rectangle(overlay, (0, 0), (width, band), (12, 22, 20), -1)
    cv2.addWeighted(overlay, 0.82, rendered, 0.18, 0, rendered)
    scale = max(0.42, min(0.8, width / 900))
    text = f"{title}  |  {timestamp:.3f}"
    cv2.putText(
        rendered,
        text[:160],
        (12, max(24, band - 11)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 250, 248),
        2,
        cv2.LINE_AA,
    )
    return rendered


def write_clip(archive, frames, name, directory):
    if len(frames) < 2:
        return
    duration = max(0.1, frames[-1][0] - frames[0][0])
    fps = min(30, max(1, (len(frames) - 1) / duration))
    height, width = frames[0][1].shape[:2]
    path = Path(directory) / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if writer.isOpened():
            for _, frame in frames:
                writer.write(cv2.resize(frame, (width, height)))
    finally:
        writer.release()
    if path.exists() and path.stat().st_size > 0:
        archive.write(path, name)


def save_bundle(frames, metadata):
    key = f"{uuid.uuid4()}.bundle"
    frames = list(frames)
    title = detection_title(metadata)
    metadata = {**metadata, "evidence_format": 2, "frame_title": title, "frame_count": len(frames)}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        review_frames = []
        for index, (timestamp, frame) in enumerate(frames):
            success, jpg = cv2.imencode(".jpg", frame)
            if success:
                archive.writestr(f"frames/{index:04d}-{timestamp:.3f}.jpg", jpg.tobytes())
            rendered = titled_frame(frame, title, timestamp)
            review_frames.append((timestamp, rendered))
            success, jpg = cv2.imencode(".jpg", rendered)
            if success:
                archive.writestr(f"review_frames/{index:04d}-{timestamp:.3f}.jpg", jpg.tobytes())
        with tempfile.TemporaryDirectory(prefix="surveilx-evidence-") as directory:
            write_clip(archive, frames, "clip.mp4", directory)
            write_clip(archive, review_frames, "review_clip.mp4", directory)
    payload = cipher().encrypt(buffer.getvalue())
    if settings.s3_bucket:
        client().put_object(Bucket=settings.s3_bucket, Key=key, Body=payload)
    else:
        directory = settings.data_dir / "evidence"
        directory.mkdir(exist_ok=True)
        (directory / key).write_bytes(payload)
    return key


def read_bundle(key):
    if "/" in key or "\\" in key or not key.endswith(".bundle"):
        raise ValueError("Invalid evidence key")
    payload = (
        client().get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read()
        if settings.s3_bucket
        else (settings.data_dir / "evidence" / key).read_bytes()
    )
    return cipher().decrypt(payload)


def delete_bundle(key):
    if settings.s3_bucket:
        client().delete_object(Bucket=settings.s3_bucket, Key=key)
    else:
        (settings.data_dir / "evidence" / key).unlink(missing_ok=True)
