import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from training.datasets import validate_manifest


def import_bundle(payload, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Dataset version already exists")
    with tempfile.TemporaryDirectory(prefix="surveilx-import-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 5000 or sum(item.file_size for item in entries) > 500 * 1024**2:
                raise ValueError("Dataset exceeds import size limit")
            for item in entries:
                path = (root / item.filename).resolve()
                if not path.is_relative_to(root) or "\\" in item.filename or ":" in item.filename:
                    raise ValueError("Unsafe archive path")
                if item.is_dir():
                    continue
                if path.suffix.lower() not in {".json", ".npz", ".png", ".jpg", ".jpeg"}:
                    raise ValueError("Dataset bundles may contain JSON, NPZ, PNG and JPEG files")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        manifest, counts = validate_manifest(root / "manifest.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, destination)
    return {"name": destination.name, "domain": manifest["domain"], "counts": counts}
