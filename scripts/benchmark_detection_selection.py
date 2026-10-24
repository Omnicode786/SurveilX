"""Bounded inference-only timing on existing images/weights. Never trains or deploys."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from surveilx.detection_preferences import DetectionSelection, infer_selected
from surveilx.vision import YOLODetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--resolution', type=int, default=320)
    args = parser.parse_args()
    if not 3 <= args.iterations <= 100:
        parser.error('Use 3..100 iterations')
    torch.set_num_threads(2)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    images = [args.dataset.parent / row['image'] for row in dataset['samples'] if row['split'] == 'validation'][:4]
    frames = [cv2.imread(str(path)) for path in images]
    if not frames or any(frame is None for frame in frames):
        raise ValueError('Existing decodable validation images required')
    model = YOLODetector(args.weights)
    labels = list(model.model.names.values())
    choices = {'all': DetectionSelection(),
        'person_only': DetectionSelection(disabled_labels=[label for label in labels if label != 'person']),
        'all_off': DetectionSelection(disabled_labels=labels)}
    forwards = []
    # Actual module forward hook proves all-off bypasses the network.
    # Hook the actual prediction backend (Ultralytics can copy/fuse the source
    # module during warmup, so hooking YOLO.model would miss real forwards).
    hook = model.model.predictor.model.register_forward_pre_hook(lambda *args: forwards.append(1))
    try:
        for selection in choices.values():
            for frame in frames:
                infer_selected(model, frame, args.resolution, selection)
        results = {name: {'ms': [], 'forwards': 0, 'detections': 0} for name in choices}
        for iteration in range(args.iterations):
            # Rotate order so no setting always benefits from the warmest GPU.
            names = list(choices)
            names = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in names:
                before = len(forwards)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                values = infer_selected(model, frames[iteration % len(frames)], args.resolution, choices[name])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                results[name]['ms'].append((time.perf_counter() - start) * 1000)
                results[name]['forwards'] += len(forwards) - before
                results[name]['detections'] += len(values)
        for row in results.values():
            row['median_ms'] = float(np.median(row['ms']))
            row['p95_ms'] = float(np.percentile(row['ms'], 95))
        if any(results[name]['forwards'] != args.iterations for name in ('all', 'person_only')) or results['all_off']['forwards']:
            raise AssertionError('Backend forward counts do not match the requested work')
        report = {'scope': 'inference only, four existing validation images, not an accuracy or live FPS benchmark',
            'device': model.device, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch': torch.__version__, 'iterations_per_setting': args.iterations, 'resolution': args.resolution,
            'weights': str(args.weights), 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'images': [str(path) for path in images], 'results': results, 'training': False, 'activation': False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({name: {key: value for key, value in row.items() if key != 'ms'} for name, row in results.items()}, indent=2))
    finally:
        hook.remove()


if __name__ == '__main__':
    main()
