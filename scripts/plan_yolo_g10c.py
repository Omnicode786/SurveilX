"""Restart in a fresh version after disk recovery, preserving verified G10 weights."""
import argparse
import json
from pathlib import Path

from scripts.train_accuracy_campaign import verify_job
from training.datasets import digest


def create(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Choose a new campaign directory')
    previous_path = Path('data/generations/yolo-g10b-commit-1024/plan.json')
    if digest(previous_path) != 'cb83d24d36a587df0a3a0593ea485ada5d511b27aadecee73ad28f4968a3448f':
        raise ValueError('Previous campaign plan changed')
    previous = json.loads(previous_path.read_text())
    code_paths = [*sorted(Path('training').glob('*.py')), Path('scripts/train_accuracy_campaign.py'),
        Path(__file__), Path('surveilx/accelerators.py'), Path('surveilx/training_runtime.py'), Path('surveilx/config.py')]
    sources = {path.as_posix(): digest(path) for path in code_paths}
    jobs = []
    for old in previous['jobs']:
        job = {**old, 'version': old['version'].replace('yolo-g10b-', 'yolo-g10c-', 1),
            'code_sha256': sources, 'options': dict(old['options'])}
        # Keep the verified G10-A best, never the disk-full G10b checkpoint.
        if (Path('data/runs') / job['version']).exists():
            raise ValueError('Refusing to overwrite an existing version')
        verify_job(job)
        jobs.append(job)
    plan = {**previous, 'name': directory.name, 'jobs': jobs,
        'predecessor': previous['name'], 'minimum_free_disk_bytes': 3 * 2**30,
        'training_min_free_disk_bytes': 512 * 2**20,
        'training_min_available_commit_bytes': 512 * 2**20,
        'limitations': [*previous['limitations'],
            'G10b failed while saving epoch-one best.pt on a full disk; its checkpoints are not initialization sources.',
            'G10c adds per-batch disk/commit guards and staged atomic checkpoint replacement.',
            'Validation improvements are provisional until independent calibration/test and acceptance checks finish.']}
    directory.mkdir(parents=True)
    target = directory / 'plan.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(create(parser.parse_args().directory))
