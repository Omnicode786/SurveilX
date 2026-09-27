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


def save_bundle(frames, metadata):
    key = f"{uuid.uuid4()}.bundle"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        for index, (timestamp, frame) in enumerate(frames):
            success, jpg = cv2.imencode(".jpg", frame)
            if success:
                archive.writestr(f"frames/{index:04d}-{timestamp:.3f}.jpg", jpg.tobytes())
        if len(frames) >= 2:
            duration = max(0.1, frames[-1][0] - frames[0][0])
            fps = min(30, max(1, (len(frames) - 1) / duration))
            height, width = frames[0][1].shape[:2]
            with tempfile.TemporaryDirectory(prefix="surveilx-evidence-") as directory:
                path = Path(directory) / "clip.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                try:
                    if writer.isOpened():
                        for _, frame in frames:
                            writer.write(cv2.resize(frame, (width, height)))
                finally:
                    writer.release()
                if path.exists() and path.stat().st_size > 0:
                    archive.write(path, "clip.mp4")
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
