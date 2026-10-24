"""Preserve the successful partial weights and lower host-memory pressure."""
import argparse
import json
from pathlib import Path

from scripts.train_accuracy_campaign import verify_job
from training.datasets import digest


def create(directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Choose a new campaign directory')
    previous_path = Path('data/generations/yolo-g10-local-1024/plan.json')
    previous = json.loads(previous_path.read_text())
    if digest(previous_path) != '5a474250fc41f48d9a933eaf895b329ea87a8f913c704c6bdd0253f160600747':
        raise ValueError('Previous campaign plan changed')
    history = json.loads(Path('data/runs/yolo-g10-weapons-local/native_history.json').read_text())
    checkpoint = Path('data/runs/yolo-g10-weapons-local/fit/weights/best.pt')
    code_paths = [*sorted(Path('training').glob('*.py')), Path('scripts/train_accuracy_campaign.py'),
        Path('scripts/plan_yolo_g10b.py'), Path('surveilx/accelerators.py'), Path('surveilx/training_runtime.py'), Path('surveilx/config.py')]
    sources = {path.as_posix(): digest(path) for path in code_paths}
    jobs = []
    for old in previous['jobs']:
        job = {**old, 'version': old['version'].replace('yolo-g10-', 'yolo-g10b-', 1),
            'code_sha256': sources, 'options': dict(old['options'])}
        options = job['options']
        if job['version'].startswith('yolo-g10b-weapons'):
            options['batch'] = 4  # accumulate 2: nominal effective batch stays eight
        if job['version'] == 'yolo-g10b-weapons-local':
            options.update({'adapter-version': 'preserve', 'epochs': 24,
                'initialize-partial': checkpoint.as_posix(), 'partial-sha256': digest(checkpoint),
                'partial-plan': previous_path.as_posix(), 'partial-plan-sha256': digest(previous_path)})
            job['reason'] = 'Continue preserved local-v2 best weights after Windows CPU allocation failure; new optimizer/schedule, not exact resume. Original G8c parent remains comparison baseline. Weapons control is now a lineage baseline, not a paired architecture ablation.'
        if (Path('data/runs') / job['version']).exists():
            raise ValueError('Refusing to overwrite an existing version')
        verify_job(job)
        jobs.append(job)
    plan = {**previous, 'name': directory.name, 'jobs': jobs, 'gpu_memory_fraction': .85,
        'minimum_free_disk_bytes': 768 * 2**20, 'minimum_available_commit_bytes': 3 * 2**30,
        'predecessor': previous['name'], 'recovered_best_validation_map50': max(r['validation_map50'] for r in history),
        'limitations': [*previous['limitations'], 'Windows commit failure stopped G10 during epoch10; nine epochs preserved.',
            'Weapons uses a preserved learned local adapter; its comparison to a fresh v1 control does not isolate architecture alone.',
            'PPE retains the paired architecture experiment. Resource reserves remain enforced.']}
    directory.mkdir(parents=True)
    target = directory / 'plan.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(create(parser.parse_args().directory))
