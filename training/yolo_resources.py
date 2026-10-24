"""Mid-run resource checks and failure-safe checkpoint replacement."""
import os
from pathlib import Path
import shutil
import tempfile

from surveilx.training_runtime import available_commit_bytes


def check_resources(directory, disk_minimum=None, commit_minimum=None):
    disk_minimum = int(os.environ.get('SURVEILX_TRAINING_MIN_DISK_BYTES', 512 * 2**20)) if disk_minimum is None else disk_minimum
    commit_minimum = int(os.environ.get('SURVEILX_TRAINING_MIN_COMMIT_BYTES', 512 * 2**20)) if commit_minimum is None else commit_minimum
    if shutil.disk_usage(str(directory)).free < disk_minimum:
        raise RuntimeError('Training resource guard: disk reserve reached; last complete checkpoint preserved')
    commit = available_commit_bytes()
    if commit is not None and commit < commit_minimum:
        raise RuntimeError('Training resource guard: Windows commit reserve reached; last complete checkpoint preserved')


def protected_save(trainer, save):
    """Write all blobs first; disk-full/serialization errors leave old files intact."""
    original = trainer.wdir, trainer.last, trainer.best
    root = Path(trainer.wdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    check_resources(root)
    staging = Path(tempfile.mkdtemp(prefix='.checkpoint-', dir=root))
    try:
        trainer.wdir, trainer.last, trainer.best = staging, staging / Path(trainer.last).name, staging / Path(trainer.best).name
        result = save()
        files = list(staging.iterdir())
        if any(not p.is_file() or p.is_symlink() or p.suffix != '.pt' for p in files):
            raise ValueError('Unexpected checkpoint staging output')
        for path in files:
            with path.open('r+b') as stream:
                os.fsync(stream.fileno())
        for path in files:
            os.replace(path, root / path.name)
        return result
    finally:
        trainer.wdir, trainer.last, trainer.best = original
        assert staging.resolve().parent == root and staging.name.startswith('.checkpoint-')
        for path in staging.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
        staging.rmdir()
