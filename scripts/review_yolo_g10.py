"""Existing-data, validation-only architecture/parent review. Never trains."""
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

from training.datasets import digest
from training.yolo_pipeline import native_validation
from training.detection_pipeline import resolve_asset


def data_summary(path, size, geometry):
    manifest = json.loads(path.read_text())
    classes = []
    samples = [row for row in manifest['samples'] if row['split'] == 'train']
    dimensions = {}
    for index, label in enumerate(manifest['classes']):
        sizes, ratios, groups = [], [], set()
        images = 0
        for row in samples:
            found = False
            if 'width' in row and 'height' in row:
                width, height = row['width'], row['height']
            else:
                if row['image'] not in dimensions:
                    with Image.open(resolve_asset(path.parent, row['image'])) as image:
                        dimensions[row['image']] = image.size
                width, height = dimensions[row['image']]
            for box, category in zip(row['boxes'], row['labels'], strict=True):
                if category != index:
                    continue
                found = True
                w, h = (box[2] - box[0]) * width, (box[3] - box[1]) * height
                factor = min(size / width, size / height)
                sizes.append(min(w * factor, h * factor) if geometry == 'letterbox' else min(w / width * size, h / height * size))
                ratios.append(max(w / h, h / w))
            if found:
                images += 1
                groups.add(row['group'])
        classes.append({'class': label, 'objects': len(sizes), 'images': images, 'recorded_groups': len(groups),
            'median_short_side_px': float(np.median(sizes)) if sizes else None,
            'under_8px': sum(x < 8 for x in sizes), 'median_elongation': float(np.median(ratios)) if ratios else None})
    return {'dataset_sha256': digest(path), 'train_images': len(samples), 'size': size, 'geometry': geometry,
            'empty_train_images': sum(not row['boxes'] for row in samples), 'classes': classes}


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for bounded review')
    torch.cuda.set_per_process_memory_fraction(.85)
    stop = json.loads(Path('data/generations/yolo-g9-focus-1022/user-stop-artifacts.json').read_text())
    for row in stop['files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError('Preserved G9 checkpoint changed')
    report = {'scope': 'validation-only parent comparison, no training or model acceptance',
        'gpu': torch.cuda.get_device_name(0), 'preserved_g9_files_verified': len(stop['files']), 'data': {}, 'weapons_parents': {}}
    for dataset, size, geometry in [('dangerous-items-development-v1', 512, 'letterbox'),
            ('sh17-development-v2', 640, 'letterbox'), ('dfire-development-v2', 512, 'stretch'), ('pennfudan', 320, 'stretch')]:
        print('Analysing training labels:', dataset, flush=True)
        report['data'][dataset] = data_summary(Path('data/datasets') / dataset / 'manifest.json', size, geometry)
    path = Path('data/datasets/dangerous-items-development-v1/manifest.json')
    manifest = json.loads(path.read_text())
    choices = {'accuracy-g3-weapons-yolo': Path('data/runs/accuracy-g3-weapons-yolo/weights.pt'),
        'capacity-g8c-weapons-yolo': Path('data/runs/capacity-g8c-weapons-yolo/weights.pt'),
        'g9-partial-best': Path('data/runs/yolo-g9-weapons-v3/fit/weights/best.pt')}
    for name, weights in choices.items():
        model = YOLO(str(weights)).model.float().to('cuda').eval()
        metrics = native_validation(model, path, manifest, 512, 'letterbox')
        report['weapons_parents'][name] = {'weights_sha256': digest(weights), 'validation': metrics}
        print(name, metrics['map50'], flush=True)
        del model
        torch.cuda.empty_cache()
    report['weapons_validation_winner'] = max(report['weapons_parents'], key=lambda key: report['weapons_parents'][key]['validation']['map50'])
    target = Path('reports/yolo-g10-design-review.json')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved', target, 'winner', report['weapons_validation_winner'], flush=True)


if __name__ == '__main__':
    main()
