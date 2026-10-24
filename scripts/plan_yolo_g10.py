"""New, immutable modified-YOLO continuation after validation/capacity review."""
import argparse
import json
from pathlib import Path

from scripts.train_accuracy_campaign import verify_job
from training.datasets import digest


def create(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Choose a fresh campaign directory')
    review_path = Path('reports/yolo-g10-design-review.json')
    capacity_path = Path('reports/yolo-g10-capacity.json')
    review, capacity = json.loads(review_path.read_text()), json.loads(capacity_path.read_text())
    if review['weapons_validation_winner'] != 'capacity-g8c-weapons-yolo':
        raise ValueError('Revisit parent choice; reviewed validation winner changed')
    probes = {row['parent']: row for row in capacity['results']}
    if probes['capacity-g8c-weapons-yolo']['max_reserved_mib'] > 1600 or probes['capacity-g8c-ppe-yolo']['max_reserved_mib'] > 1600:
        raise ValueError('Reviewed batch exceeds VRAM headroom')
    # Paired architecture controls share parents, settings, seeds and exact data.
    recipes = [
        ('weapons-local', 'dangerous-items-development-v1', 'capacity-g8c-weapons-yolo', 512, 8, .0001, 'letterbox', 'local-v2', 32, 8, 0.),
        ('weapons-control', 'dangerous-items-development-v1', 'capacity-g8c-weapons-yolo', 512, 8, .0001, 'letterbox', 'preserve', 32, 8, 0.),
        ('ppe-local', 'sh17-development-v2', 'capacity-g8c-ppe-yolo', 640, 2, .0001, 'letterbox', 'local-v2', 32, 8, .35),
        ('ppe-control', 'sh17-development-v2', 'capacity-g8c-ppe-yolo', 640, 2, .0001, 'letterbox', 'preserve', 32, 8, .35),
        ('fire-local', 'dfire-development-v2', 'yolo-g9-fire-control', 512, 4, .0001, 'stretch', 'local-v2', 24, 8, 0.),
        ('person-retention', 'pennfudan', 'accuracy-g2-person-yolo', 320, 8, .00005, 'stretch', 'preserve', 8, 4, 0.),
    ]
    code_paths = [*sorted(Path('training').glob('*.py')), Path('scripts/train_accuracy_campaign.py'),
        Path('scripts/plan_yolo_g10.py'), Path('surveilx/accelerators.py'), Path('surveilx/training_runtime.py'), Path('surveilx/config.py')]
    code = {path.as_posix(): digest(path) for path in code_paths}
    jobs = []
    for suffix, dataset, parent, size, batch, rate, geometry, architecture, epochs, patience, balance in recipes:
        manifest = Path('data/datasets') / dataset / 'manifest.json'
        weights = Path('data/runs') / parent / 'weights.pt'
        version = f'yolo-g10-{suffix}'
        if (Path('data/runs') / version).exists():
            raise ValueError('Refusing to overwrite an existing run')
        job = {'version': version, 'pipeline': 'yolo', 'dataset': dataset, 'parent': parent,
            'dataset_sha256': digest(manifest), 'parent_sha256': digest(weights), 'code_sha256': code,
            'options': {'epochs': epochs, 'patience': patience, 'threads': 3, 'seed': 42,
                'imgsz': size, 'batch': batch, 'learning-rate': rate, 'preprocessing': geometry,
                'training-profile': 'finetune-v2', 'effective-batch-size': 8,
                'adapter-version': architecture, 'class-weights-power': balance},
            'reason': 'New preserved-weight continuation, not optimizer resume; paired architecture controls for weapons/PPE. Validation selection, matched-profile parent evaluation, separate calibration/test, no activation.'}
        verify_job(job)
        jobs.append(job)
    plan = {'name': directory.name, 'required_cuda': True, 'gpu_memory_fraction': .9,
        'minimum_free_disk_bytes': 512 * 2**20, 'minimum_available_ram_bytes': 1 * 2**30,
        'minimum_available_commit_bytes': 2 * 2**30, 'minimum_free_gpu_mib': 500,
        'selection': 'Validation only; same-profile parent calibration/test reporting after selection. No automatic acceptance or activation.',
        'excluded': ['scratch/SVA', 'temporal-event training', 'new datasets', 'automatic model activation'],
        'design_review_sha256': digest(review_path), 'capacity_sha256': digest(capacity_path),
        'limitations': ['No perfect accuracy guarantee. Rare PPE labels and source diversity remain limiting.',
            'Fire/person are bounded continuation/retention checks, not paired architecture ablations.',
            'GPU, disk, physical RAM and Windows commit guards may stop further launches.'],
        'jobs': jobs}
    directory.mkdir(parents=True)
    target = directory / 'plan.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(create(parser.parse_args().directory))
