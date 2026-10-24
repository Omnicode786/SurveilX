"""Fail-closed provenance for weight-only continuation of interrupted YOLO runs."""
import json
from pathlib import Path

from training.datasets import digest


def verify_partial(checkpoint, checkpoint_sha256, plan_path, plan_sha256, dataset_sha256, parent_sha256):
    checkpoint, plan_path = Path(checkpoint).resolve(), Path(plan_path).resolve()
    if not checkpoint_sha256 or digest(checkpoint) != checkpoint_sha256 or digest(plan_path) != plan_sha256:
        raise ValueError('Partial checkpoint or source plan checksum mismatch')
    run = checkpoint.parent.parent.parent
    if checkpoint.name not in {'best.pt', 'last.pt'} or checkpoint.parent != run / 'fit/weights':
        raise ValueError('Partial initialization requires a known fit/weights checkpoint')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    matches = [job for job in plan['jobs'] if job['version'] == run.name and job['pipeline'] == 'yolo']
    if len(matches) != 1:
        raise ValueError('Partial run is not uniquely bound to its source plan')
    job = matches[0]
    if job['dataset_sha256'] != dataset_sha256 or job['parent_sha256'] != parent_sha256:
        raise ValueError('Partial run dataset/parent lineage differs from the requested continuation')
    prepared = json.loads((run / 'dataset/prepared-integrity.json').read_text(encoding='utf-8'))
    if prepared['dataset_sha256'] != dataset_sha256:
        raise ValueError('Partial prepared-data provenance differs from the dataset')
    for name, sha in prepared['sources'].items():
        if job['code_sha256'].get('training/' + name) != sha:
            raise ValueError('Partial training sources differ from the source plan')
    return {'checkpoint': str(checkpoint), 'weights_sha256': checkpoint_sha256,
        'source_plan': str(plan_path), 'source_plan_sha256': plan_sha256,
        'dataset_sha256': dataset_sha256, 'parent_sha256': parent_sha256,
        'optimizer_restored': False, 'scope': 'Preserved weights only; new optimizer/schedule/version'}
