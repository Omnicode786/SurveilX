import json

import pytest

from training.datasets import digest
from training.yolo_initialization import verify_partial


def test_partial_weights_require_bound_plan_dataset_parent_and_source(tmp_path):
    checkpoint = tmp_path / 'partial/fit/weights/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'preserved weights')
    prepared = tmp_path / 'partial/dataset/prepared-integrity.json'
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({'dataset_sha256': 'data', 'sources': {'yolo_model.py': 'source'}}))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'jobs': [{'version': 'partial', 'pipeline': 'yolo', 'dataset_sha256': 'data',
        'parent_sha256': 'parent', 'code_sha256': {'training/yolo_model.py': 'source'}}]}))
    args = [checkpoint, digest(checkpoint), plan, digest(plan), 'data', 'parent']
    assert verify_partial(*args)['optimizer_restored'] is False
    with pytest.raises(ValueError, match='lineage'):
        verify_partial(*args[:-1], 'foreign-parent')
    prepared.write_text(json.dumps({'dataset_sha256': 'data', 'sources': {'yolo_model.py': 'foreign-source'}}))
    with pytest.raises(ValueError, match='sources'):
        verify_partial(*args)
    checkpoint.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='checksum'):
        verify_partial(*args)
