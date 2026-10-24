"""Bounded forward/backward capacity check; no optimizer, saved weights or training run."""
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from training.detection_geometry import prepare_image, transform_boxes
from training.detection_pipeline import resolve_asset
from training.yolo_model import upgrade_adapters


def main():
    torch.set_num_threads(3)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(.9)
    results = []
    for parent, dataset, size, count, geometry in [
        ('capacity-g8c-weapons-yolo', 'dangerous-items-development-v1', 512, 8, 'letterbox'),
        ('capacity-g8c-ppe-yolo', 'sh17-development-v2', 640, 2, 'letterbox')]:
        path = Path('data/datasets') / dataset / 'manifest.json'
        data = json.loads(path.read_text())
        sample = max((row for row in data['samples'] if row['split'] == 'train'), key=lambda row: len(row['labels']))
        pixels = cv2.imread(str(resolve_asset(path.parent, sample['image'])))
        rgb, transform = prepare_image(pixels, size, geometry)
        boxes = transform_boxes(np.asarray(sample['boxes'], dtype=np.float32), transform)
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, :2] += boxes[:, 2:] / 2
        batch = {'img': torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].repeat(count, 1, 1, 1).cuda() / 255,
            'bboxes': torch.from_numpy(boxes).repeat(count, 1).cuda(),
            'cls': torch.tensor(sample['labels'], dtype=torch.float32).repeat(count)[:, None].cuda(),
            'batch_idx': torch.arange(count).repeat_interleave(len(boxes)).cuda()}
        model = YOLO(str(Path('data/runs') / parent / 'weights.pt')).model.float().cuda()
        before = sum(p.numel() for p in model.parameters())
        upgrade_adapters(model, 'local-v2')
        model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **model.args})
        model.requires_grad_(True).train()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _ = model(batch)
            loss.sum().backward()
            torch.cuda.synchronize()
            if not torch.isfinite(loss).all():
                raise ValueError('Non-finite probe loss')
            if iteration:
                times.append(time.perf_counter() - start)
            del loss
        results.append({'parent': parent, 'resolution': size, 'batch': count,
            'labels_per_image': len(boxes), 'additional_parameters': sum(p.numel() for p in model.parameters()) - before,
            'max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'max_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'median_forward_backward_seconds': float(np.median(times)), 'optimizer_steps': 0})
        print(json.dumps(results[-1]), flush=True)
        del model, batch
        gc.collect()
        torch.cuda.empty_cache()
    report = {'gpu': torch.cuda.get_device_name(0), 'scope': 'capacity only; repeated dense train sample, no optimizer steps or persisted weights',
        'results': results, 'accuracy_claim': False}
    Path('reports/yolo-g10-capacity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
