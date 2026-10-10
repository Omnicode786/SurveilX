"""Read-only g4 frozen dataset / historical SVA-v1 assignment audit. No training."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ('train', 'validation', 'calibration', 'test')
DATASETS = [('dfire-development-v1', 384), ('dfire-development-v2', 384),
            ('dangerous-items-development-v1', 384), ('sh17-development-v2', 512)]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summary(values):
    if not len(values):
        return None
    return dict(zip(('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'),
                    map(float, np.percentile(values, [0, 10, 25, 50, 75, 90, 100])), strict=True))


def grid(image_size):
    points, steps, levels = [], [], []
    for level, stride in enumerate((8, 16, 32)):
        size = (image_size + stride - 1) // stride
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        points.append(np.stack(((xx.ravel() + .5) / size, (yy.ravel() + .5) / size), -1))
        steps.append(np.full((size * size, 2), 1 / size))
        levels.append(np.full(size * size, level))
    return np.concatenate(points).astype(np.float32), np.concatenate(steps).astype(np.float32), np.concatenate(levels)


def historical_assignment(boxes, image_size):
    """Reproduce v1 center-window / longest-side scale / smallest-area conflicts."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    points, steps, levels = grid(image_size)
    if not len(boxes):
        return dict(counts=[], desired_levels=[], candidate_counts=[], outside_box_counts=[],
                    point_winners=np.full(len(points), -1))
    sizes, centers = boxes[:, 2:] - boxes[:, :2], (boxes[:, :2] + boxes[:, 2:]) / 2
    desired = np.clip(np.floor(np.log2(np.maximum(sizes.max(-1), 1e-6) / .16)).astype(int), 0, 2)
    distance = np.abs((points[:, None] - centers[None]) / steps[:, None]).max(-1)
    eligible = (distance <= 1.5) & (levels[:, None] == desired[None])
    for index in range(len(boxes)):
        if not eligible[:, index].any():
            eligible[np.where(levels == desired[index], distance[:, index], np.inf).argmin(), index] = True
    cost = np.where(eligible, sizes.prod(-1)[None], np.inf)
    winner = cost.argmin(-1)
    winner[~np.isfinite(cost.min(-1))] = -1
    counts, outside = [], []
    for index, box in enumerate(boxes):
        selected = points[winner == index]
        counts.append(len(selected))
        outside.append(int(np.sum(((selected < box[:2]) | (selected > box[2:])).any(-1))))
    return dict(counts=counts, desired_levels=desired.tolist(), candidate_counts=eligible.sum(0).tolist(),
                outside_box_counts=outside, point_winners=winner)


def describe(rows):
    return {'objects': len(rows),
            'short_side_pixels': summary([r['short'] for r in rows]),
            'long_side_pixels': summary([r['long'] for r in rows]),
            'aspect_ratio_after_square_stretch': summary([r['aspect'] for r in rows]),
            'aspect_ratio_native': summary([r['native_aspect'] for r in rows]),
            'letterbox_short_side_pixels_at_same_size': summary([r['letterbox_short'] for r in rows]),
            'tiny_short_side_under_8px': sum(r['short'] < 8 for r in rows),
            'short_side_under_16px': sum(r['short'] < 16 for r in rows),
            'area_under_32_squared_px': sum(r['area'] < 32 ** 2 for r in rows),
            'elongated_aspect_over_4': sum(r['aspect'] > 4 for r in rows),
            'assignments': {'positive_count': sum(r['positives'] for r in rows),
                            'positive_counts_per_object': summary([r['positives'] for r in rows]),
                            'positive_counts_per_level': [sum(r['positives'] for r in rows if r['level'] == i) for i in range(3)],
                            'objects_lost_to_conflicts': sum(r['positives'] == 0 for r in rows),
                            'objects_with_fewer_than_three_positives': sum(r['positives'] < 3 for r in rows),
                            'positive_points_outside_assigned_box': sum(r['outside'] for r in rows)}}


def analyze_dataset(directory, image_size):
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    classes = manifest['classes']
    rows, splits, groups, hashes, sources = [], Counter(), defaultdict(Counter), defaultdict(list), defaultdict(list)
    empty, images, distortions = Counter(), {s: Counter() for s in SPLITS}, defaultdict(list)
    bytes_verified = 0
    for sample in manifest['samples']:
        split = sample['split']
        splits[split] += 1
        groups[split][sample['group']] += 1
        images[split].update(set(sample['labels']))
        empty[split] += int(not sample['boxes'])
        image_path = (directory / sample['image']).resolve()
        if not image_path.is_relative_to(directory.resolve()):
            raise ValueError('Image escaped dataset directory')
        actual = digest(image_path)
        if actual != sample['sha256']:
            raise ValueError(f'Image checksum changed: {image_path}')
        bytes_verified += image_path.stat().st_size
        hashes[actual].append(split)
        sources[sample.get('source_sha256') or sample.get('source_member') or actual].append(split)
        with Image.open(image_path) as image:
            width, height = image.size
        distortions[split].append(max(width / height, height / width))
        assignment = historical_assignment(sample['boxes'], image_size)
        for i, (box, label) in enumerate(zip(sample['boxes'], sample['labels'], strict=True)):
            wh = np.array([box[2] - box[0], box[3] - box[1]])
            pixels, native = wh * image_size, wh * [width, height]
            letterbox = native * image_size / max(width, height)
            rows.append(dict(split=split, label=label, short=float(pixels.min()), long=float(pixels.max()),
                             area=float(pixels.prod()), aspect=float(pixels.max() / pixels.min()),
                             native_aspect=float(native.max() / native.min()), letterbox_short=float(letterbox.min()),
                             positives=assignment['counts'][i], outside=assignment['outside_box_counts'][i],
                             level=assignment['desired_levels'][i]))
    per_split = {}
    for split in SPLITS:
        per_split[split] = dict(images=splits[split], empty_images=empty[split], recorded_groups=len(groups[split]),
                               group_image_counts=dict(groups[split]), image_stretch_anisotropy=summary(distortions[split]),
                               fraction_images_stretched_over_1_5x=float(np.mean(np.array(distortions[split]) > 1.5)),
                               **describe([r for r in rows if r['split'] == split]))
    per_class = [dict(class_name=name, class_index=i, splits={split: dict(images=images[split][i],
                 **describe([r for r in rows if r['split'] == split and r['label'] == i])) for split in SPLITS})
                 for i, name in enumerate(classes)]
    counts = [r['splits']['train']['objects'] for r in per_class]
    image_counts = np.array([images['train'][i] for i in range(len(classes))])
    factors = np.minimum(np.sqrt(max(int(image_counts.max()), 1) / np.maximum(image_counts, 1)), 4.)
    train = [s for s in manifest['samples'] if s['split'] == 'train']
    weights = np.array([max((factors[i] for i in s['labels']), default=1.) for s in train])
    probabilities = weights / weights.sum()
    for i, row in enumerate(per_class):
        row['expected_sampled_images_per_epoch'] = float(len(train) * sum(p for p, s in zip(probabilities, train, strict=True) if i in s['labels']))
        row['expected_sampled_instances_per_epoch'] = float(len(train) * sum(p * s['labels'].count(i) for p, s in zip(probabilities, train, strict=True)))
    return dict(dataset=directory.name, manifest_sha256=digest(path), image_size=image_size,
                preprocessing='square-stretch-v1', verified_image_bytes=bytes_verified, class_count=len(classes),
                splits=per_split, classes=per_class, train_instance_imbalance_max_to_min_nonzero=max(counts) / min(c for c in counts if c),
                train_classes_without_labels=[classes[i] for i, c in enumerate(counts) if not c],
                exact_duplicate_content_records=sum(len(v) - 1 for v in hashes.values()),
                exact_content_cross_split_duplicates=sum(len(set(v)) > 1 for v in hashes.values()),
                repeated_source_identifiers=sum(len(v) - 1 for v in sources.values()),
                source_identifiers_cross_split=sum(len(set(v)) > 1 for v in sources.values()),
                provenance=manifest.get('provenance', {}))


def compare_expansion(base):
    old = json.loads((base / 'dfire-development-v1/manifest.json').read_text())
    new = json.loads((base / 'dfire-development-v2/manifest.json').read_text())
    old_records = {s['sha256']: s for s in old['samples']}
    new_records = {s['sha256']: s for s in new['samples']}
    added = [s for key, s in new_records.items() if key not in old_records]
    return dict(all_old_records_retained_identically=all(new_records.get(k) == v for k, v in old_records.items()),
                old_record_count=len(old_records), additions_by_split=dict(Counter(s['split'] for s in added)),
                interpretation='Only fire received additional images in g4b. Data, focal loss, batch size and training device changed together; focal effect is not isolated.')


def cleanup_candidates(root):
    sources = {digest(p): p for p in (root / 'data/datasets').glob('*/manifest.json')}
    rows = []
    for run in sorted((root / 'data/runs').iterdir()):
        directory, artifact = run / 'dataset', run / 'manifest.json'
        if not directory.is_dir() or not artifact.is_file():
            continue
        model = json.loads(artifact.read_text())
        source = sources.get(model.get('dataset_sha256'))
        if source is None or model.get('stage') != 'candidate':
            continue
        files = [p for p in directory.rglob('*') if p.is_file()]
        if files:
            rows.append(dict(directory=str(directory.resolve()), bytes=sum(p.stat().st_size for p in files), files=len(files),
                             source_manifest=str(source.resolve()), source_manifest_sha256=model['dataset_sha256'],
                             completed_candidate=True,
                             weights_sha256_matches=(run / 'weights.pt').is_file() and digest(run / 'weights.pt') == model.get('weights_sha256'),
                             reason='Rebuildable YOLO resized-image and label materialization; source retained. Keep fit, weights, manifests, histories, predictions and acquisition archives.'))
    return dict(candidates=rows, bytes_total=sum(r['bytes'] for r in rows), deleted=False,
                warning='Check that no training/resume process uses these directories before removal. Original dataset versions remain referenced by historical artifacts; retain them.')


def markdown(report):
    lines = ['# G4 dataset and historical assignment comparison', '', report['scope'], '',
             'All image SHA256 values checked against manifests. No training or downloads.', '',
             'Sizes use actual g4 square stretching. Sub-stride size is a risk indicator, not a detection impossibility. Letterboxing preserves shape but can reduce small-object pixels at equal canvas size.', '']
    for ds in report['datasets']:
        train = ds['splits']['train']
        lines += [f"## {ds['dataset']} ({ds['image_size']} px)", '',
                  f"Train: {train['images']} images, {train['empty_images']} empty, {train['objects']} objects; {ds['class_count']} classes, {train['recorded_groups']} recorded groups. Object imbalance max/min: {ds['train_instance_imbalance_max_to_min_nonzero']:.2f}.", '',
                  '| Class | Train objects / images | Val / cal / test objects | Median short side px | <8 px | Aspect >4 | Lost objects | Positive assignments |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in ds['classes']:
            value = row['splits']['train']
            size = value['short_side_pixels']
            median = f"{size['median']:.2f}" if size else 'n/a'
            held = ' / '.join(str(row['splits'][s]['objects']) for s in SPLITS[1:])
            lines.append(f"| {row['class_name']} | {value['objects']} / {value['images']} | {held} | {median} | {value['tiny_short_side_under_8px']} | {value['elongated_aspect_over_4']} | {value['assignments']['objects_lost_to_conflicts']} | {value['assignments']['positive_count']} |")
        lines += ['', f"Assignment totals: {json.dumps(train['assignments'])}", '']
    lines += ['## Fire expansion control', '', json.dumps(report['fire_expansion'], indent=2), '',
              '## Storage cleanup candidates', '', report['cleanup']['warning'], '',
              f"Rebuildable bytes: {report['cleanup']['bytes_total']:,}. This script deletes nothing.", '']
    for row in report['cleanup']['candidates']:
        lines.append(f"- `{row['directory']}`: {row['bytes']:,} bytes; weights hash valid: {row['weights_sha256_matches']}.")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('reports/g4-dataset-analysis'))
    args = parser.parse_args()
    report = dict(schema_version=1, scope='Descriptive frozen-data and SVA-v1 assignment audit; correlation does not establish causation or independent field accuracy.',
                  script_sha256=digest(__file__), datasets=[], fire_expansion=compare_expansion(args.root / 'data/datasets'))
    for name, size in DATASETS:
        result = analyze_dataset(args.root / 'data/datasets' / name, size)
        report['datasets'].append(result)
        print(f"Verified {name}: {result['splits']['train']['images']} training images", flush=True)
    report['cleanup'] = cleanup_candidates(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    args.output.with_suffix('.md').write_text(markdown(report), encoding='utf-8')


if __name__ == '__main__':
    main()
