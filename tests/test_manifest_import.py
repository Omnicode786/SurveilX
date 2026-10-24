"""The CPU campaign controller must not load PyTorch just to verify data."""
import json
from pathlib import Path
import subprocess
import sys


def test_detection_integrity_validation_does_not_import_torch(tmp_path):
    samples = []
    for i, split in enumerate(('train', 'validation', 'calibration', 'test')):
        image = tmp_path / (split + '.png')
        # Integrity validation checks existing assets/hashes, not image decoding.
        image.write_bytes(bytes([i]))
        samples.append({'image': image.name, 'split': split, 'group': split,
            'boxes': [[.1, .1, .2, .2]], 'labels': [0]})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'task': 'detection', 'domain': 'test',
        'license': 'synthetic test fixture', 'classes': ['person'], 'samples': samples}))
    code = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('Controller attempted to import torch')
sys.meta_path.insert(0, NoTorch())
from scripts.train_accuracy_campaign import verify_job
from training.datasets import validate_manifest
manifest, counts = validate_manifest(sys.argv[1])
assert sum(counts.values()) == 4
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(manifest)], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
