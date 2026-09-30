"""Resumable bounded acquisition of explicitly catalogued public dataset archives."""

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from training.datasets import digest

SOURCES = {
    "dfire": {
        "page": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "url": "https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo",
        "license": "Author collection: CC0-1.0; authors disclaim underlying image copyrights; preserve provenance",
        "classes": ["smoke", "fire"],
        "max_bytes": 4 * 1024**3,
    },
    "dangerous-items": {
        "page": "https://zenodo.org/records/13786228",
        "url": "https://zenodo.org/api/records/13786228/files/Dangerous%20Items.zip/content",
        "license": "CC BY 4.0; Zenodo record 13786228",
        "classes": ["baseball_bat", "gun", "knife", "machete", "rifle"],
        "max_bytes": 512 * 1024**2,
    },
}


def verify_and_publish(partial, archive, root, record, offset, total, etag):
    with zipfile.ZipFile(partial) as bundle:
        inventory = [{"name": i.filename, "bytes": i.file_size, "crc": i.CRC} for i in bundle.infolist()]
    (root / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    checksum = digest(partial)
    record(
        state="verified_download",
        bytes=offset,
        total=total,
        etag=etag,
        sha256=checksum,
        entries=len(inventory),
        training_ready=False,
        next_step="Audit source groups, labels and split provenance before creating training manifests",
    )
    partial.replace(archive)
    print(json.dumps({"archive": str(archive), "sha256": checksum, "entries": len(inventory)}), flush=True)


def acquire(name):
    source = SOURCES[name]
    root = Path("data/downloads") / name
    root.mkdir(parents=True, exist_ok=True)
    archive, partial = root / "dataset.zip", root / "dataset.zip.part"
    status = root / "acquisition.json"

    def record(**extra):
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps({**source, "updated": time.time(), **extra}, indent=2), encoding="utf-8")
        temporary.replace(status)

    if archive.exists():
        previous = json.loads(status.read_text(encoding="utf-8"))
        if digest(archive) != previous.get("sha256"):
            raise ValueError("Existing archive does not match its recorded checksum")
        print("Verified existing archive", archive, flush=True)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SurveilX-DataAcquisition/1.0"}
    previous = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
    if offset and offset == previous.get("total"):
        verify_and_publish(partial, archive, root, record, offset, offset, previous.get("etag"))
        return
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if previous.get("etag"):
            headers["If-Range"] = previous["etag"]
    try:
        with urllib.request.urlopen(urllib.request.Request(source["url"], headers=headers), timeout=60) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Server cannot safely resume this partial archive; preserve it and investigate")
            etag = response.headers.get("ETag")
            if offset and previous.get("etag") and previous["etag"] != etag:
                raise ValueError("Remote archive changed during resumed acquisition")
            total = offset + int(response.headers.get("Content-Length", "0"))
            if not offset < total <= source["max_bytes"]:
                raise ValueError("Missing or oversized archive length")
            if shutil.disk_usage(root).free < (total-offset) + 3*1024**3:
                raise ValueError("Insufficient free space while retaining a 3 GiB reserve")
            record(state="downloading", bytes=offset, total=total, etag=etag)
            last = time.monotonic()
            with partial.open("ab" if offset else "wb") as target:
                while block := response.read(1024*1024):
                    offset += len(block)
                    if offset > total:
                        raise ValueError("Archive exceeded advertised size")
                    target.write(block)
                    if time.monotonic()-last >= 10:
                        target.flush()
                        record(state="downloading", bytes=offset, total=total, etag=etag)
                        print(f"{name}: {offset}/{total} bytes ({100*offset/total:.1f}%)", flush=True)
                        last = time.monotonic()
            if offset != total:
                raise ValueError("Incomplete archive; rerun to resume")
        verify_and_publish(partial, archive, root, record, offset, total, etag)
    except Exception as exc:
        record(state="interrupted", bytes=partial.stat().st_size if partial.exists() else 0,
               etag=locals().get("etag", previous.get("etag")), error=str(exc))
        raise


def acquire_with_retries(name, attempts=8):
    """Resume only cleanly truncated responses while requiring forward progress."""
    partial = Path("data/downloads") / name / "dataset.zip.part"
    for attempt in range(1, attempts + 1):
        before = partial.stat().st_size if partial.exists() else 0
        try:
            acquire(name)
            return
        except ValueError as exc:
            after = partial.stat().st_size if partial.exists() else 0
            if str(exc) != "Incomplete archive; rerun to resume" or after <= before or attempt == attempts:
                raise
            print(f"Connection ended after {after - before} bytes; resuming ({attempt}/{attempts})", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 32:
        parser.error("attempts must be 1..32")
    acquire_with_retries(args.dataset, args.attempts)
