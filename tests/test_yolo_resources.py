from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest

from training import yolo_resources as resources


def test_resource_guard_checks_disk_and_commit_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(RuntimeError, match='disk reserve'):
        resources.check_resources(tmp_path, disk_minimum=2, commit_minimum=0)
    monkeypatch.setattr(resources.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(100, 0, 100))
    monkeypatch.setattr(resources, 'available_commit_bytes', lambda: 1)
    with pytest.raises(RuntimeError, match='commit reserve'):
        resources.check_resources(tmp_path, disk_minimum=2, commit_minimum=2)


def test_failed_checkpoint_write_preserves_old_best_and_last(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, 'check_resources', lambda _: None)
    trainer = SimpleNamespace(wdir=tmp_path, last=tmp_path/'last.pt', best=tmp_path/'best.pt')
    trainer.last.write_bytes(b'old-last')
    trainer.best.write_bytes(b'old-best')
    def full_disk():
        trainer.last.write_bytes(b'new-last')
        trainer.best.write_bytes(b'partial')
        raise OSError('No space left on device')
    with pytest.raises(OSError):
        resources.protected_save(trainer, full_disk)
    assert trainer.last.read_bytes() == b'old-last' and trainer.best.read_bytes() == b'old-best'
    assert not list(tmp_path.glob('.checkpoint-*'))
    def success():
        trainer.last.write_bytes(b'new-last')
        trainer.best.write_bytes(b'new-best')
        return True
    assert resources.protected_save(trainer, success)
    assert trainer.last.read_bytes() == b'new-last' and trainer.best.read_bytes() == b'new-best'
    assert trainer.wdir == tmp_path and isinstance(trainer.best, Path)
