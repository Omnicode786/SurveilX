"""Resumable bounded download of the pinned official Windows CUDA torch wheel."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

from pip._internal.network.session import PipSession

from training.datasets import digest


URL = "https://download-r2.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp312-cp312-win_amd64.whl"
SHA256 = "11a492e40dfd18597fdb8271879a3a09496a23140deb81a6b5a005d9cff7312b"
SIZE = 2602771598
OUTPUT = Path("data/downloads/cuda/torch-2.14.0+cu126-cp312-cp312-win_amd64.whl")


def run(prefix=None):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if digest(OUTPUT) != SHA256:
            raise ValueError("Existing CUDA wheel failed publisher checksum")
        return OUTPUT
    partial, journal = OUTPUT.with_suffix(".part"), OUTPUT.with_suffix(".json")
    chunk_size = 16 * 1024**2
    if journal.exists():
        state = json.loads(journal.read_text())
        if state["url"] != URL or state["sha256"] != SHA256 or state["size"] != SIZE:
            raise ValueError("Existing download plan differs from the pinned wheel")
        if not partial.is_file() or partial.stat().st_size != SIZE:
            raise ValueError("Partial wheel is missing or has changed size")
    else:
        if partial.exists():
            raise ValueError("Unjournaled partial preserved; inspect before resuming")
        prefix = Path(prefix) if prefix else None
        length = prefix.stat().st_size if prefix else 0
        if length >= SIZE or shutil.disk_usage(OUTPUT.parent).free < SIZE + 3 * 1024**3:
            raise ValueError("Invalid prefix or insufficient disk reserve")
        with partial.open("wb") as output:
            if prefix:
                with prefix.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            output.truncate(SIZE)
        state = {
            "url": URL,
            "sha256": SHA256,
            "size": SIZE,
            "prefix": length,
            "prefix_sha256": digest(prefix) if prefix else None,
            "completed": {},
        }
        journal.write_text(json.dumps(state, indent=2))
    ranges = [(start, min(SIZE, start + chunk_size)) for start in range(state["prefix"], SIZE, chunk_size)]
    with partial.open("rb") as source:
        prefix_hash = hashlib.sha256()
        remaining = state["prefix"]
        while remaining:
            value = source.read(min(remaining, 1024 * 1024))
            prefix_hash.update(value)
            remaining -= len(value)
        if state["prefix"] and prefix_hash.hexdigest() != state["prefix_sha256"]:
            raise ValueError("Preserved download prefix changed")
        for start, end in ranges:
            source.seek(start)
            if str(start) in state["completed"]:
                if hashlib.sha256(source.read(end - start)).hexdigest() != state["completed"][str(start)]:
                    raise ValueError("Completed range checksum changed")
    local, lock = threading.local(), threading.Lock()

    def download(bounds):
        start, end = bounds
        if not hasattr(local, "session"):
            local.session = PipSession()
        for attempt in range(4):
            try:
                with local.session.get(
                    URL,
                    headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{SIZE}"
                    ):
                        raise ValueError("Publisher did not honor the exact requested byte range")
                    content = response.raw.read(end - start + 1)
                if len(content) != end - start:
                    raise ValueError("Truncated publisher range")
                with partial.open("r+b") as output:
                    output.seek(start)
                    output.write(content)
                    output.flush()
                with lock:
                    state["completed"][str(start)] = hashlib.sha256(content).hexdigest()
                    pending = journal.with_suffix(".tmp")
                    pending.write_text(json.dumps(state, indent=2))
                    pending.replace(journal)
                return end - start
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    pending = [bounds for bounds in ranges if str(bounds[0]) not in state["completed"]]
    completed = SIZE - sum(end - start for start, end in pending)
    print(
        json.dumps({"downloaded_mib": round(completed / 1024**2, 1), "total_mib": round(SIZE / 1024**2, 1)}),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(download, bounds) for bounds in pending]):
            completed += future.result()
            print(
                json.dumps(
                    {"downloaded_mib": round(completed / 1024**2, 1), "total_mib": round(SIZE / 1024**2, 1)}
                ),
                flush=True,
            )
    if digest(partial) != SHA256:
        raise ValueError("Complete CUDA wheel failed publisher SHA256; installation forbidden")
    partial.replace(OUTPUT)
    state.update(state="verified", finished=time.time())
    journal.write_text(json.dumps(state, indent=2))
    return OUTPUT


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix")
    print(run(parser.parse_args().prefix))
