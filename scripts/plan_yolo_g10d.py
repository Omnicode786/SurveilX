"""Fresh low-commit campaign after an instrumented batch-two CUDA capacity check."""
import argparse
import json
from pathlib import Path

from scripts.train_accuracy_campaign import verify_job
from training.datasets import digest


def create(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Choose a new campaign directory')
    previous_path = Path('data/generations/yolo-g10c-guarded-1024/plan.json')
    if digest(previous_path) != 'beb3e2c86d5edfbdd9bbb198c394996a1e3d43f94343a85cce2980302fe13a00':
        raise ValueError('Previous campaign plan changed')
    previous = json.loads(previous_path.read_text())
    code_paths = [*sorted(Path('training').glob('*.py')), Path('scripts/train_accuracy_campaign.py'),
        Path(__file__), Path('surveilx/accelerators.py'), Path('surveilx/training_runtime.py'), Path('surveilx/config.py')]
    sources = {path.as_posix(): digest(path) for path in code_paths}
    jobs = []
    for old in previous['jobs']:
        job = {**old, 'version': old['version'].replace('yolo-g10c-', 'yolo-g10d-', 1),
            'code_sha256': sources, 'options': dict(old['options'])}
        job['options']['batch'] = 1 if '-ppe-' in job['version'] else 4 if '-person-' in job['version'] else 2
        if (Path('data/runs') / job['version']).exists():
            raise ValueError('Refusing to overwrite an existing version')
        verify_job(job)
        jobs.append(job)
    plan = {**previous, 'name': directory.name, 'jobs': jobs, 'predecessor': previous['name'],
        'minimum_available_commit_bytes': int(3.5 * 2**30),
        'commit_probe_sha256': digest('data/generations/yolo-g10c-guarded-1024/commit-probe.json'),
        'limitations': [*previous['limitations'],
            'G10c guard stopped after its first CUDA batch; no completed epoch or checkpoint.',
            'Physical batches reduced: weapons/fire two, PPE one, person four; nominal effective eight retained.',
            'Gradient accumulation preserves nominal batch size but does not replicate larger-batch BatchNorm statistics.',
            'A resource failure now halts the entire queue even if memory recovers after child exit.',
            'Batch-two capacity probe left about 1.3 GiB commit headroom but excludes full trainer/evaluation overhead.']}
    directory.mkdir(parents=True)
    target = directory / 'plan.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(create(parser.parse_args().directory))
