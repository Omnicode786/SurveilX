"""Bounded GPU capacity diagnostic including Windows commit; never steps an optimizer."""
import json
from pathlib import Path
import threading
import time

from surveilx.training_runtime import available_commit_bytes


def main():
    import psutil
    import shutil
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            samples.append({'time': time.time(), 'commit': available_commit_bytes(),
                'ram': psutil.virtual_memory().available, 'disk': shutil.disk_usage('.').free})
            stop.wait(.25)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    results = []
    try:
        import cv2
        import numpy as np
        import torch
        from types import SimpleNamespace
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from training.detection_geometry import prepare_image, transform_boxes
        from training.detection_pipeline import resolve_asset
        torch.set_num_threads(3)
        cv2.setNumThreads(2)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.85)
        path = Path('data/datasets/dangerous-items-development-v1/manifest.json')
        data = json.loads(path.read_text())
        row = max((r for r in data['samples'] if r['split'] == 'train'), key=lambda r: len(r['labels']))
        pixels = cv2.imread(str(resolve_asset(path.parent, row['image'])))
        rgb, transform = prepare_image(pixels, 512, 'letterbox')
        boxes = transform_boxes(np.asarray(row['boxes'], dtype=np.float32), transform)
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, :2] += boxes[:, 2:] / 2
        model = YOLO('data/runs/yolo-g10-weapons-local/fit/weights/best.pt').model.float().cuda()
        model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **model.args})
        model.requires_grad_(True).train()
        for count in (2, 1):
            batch = {'img': torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].repeat(count, 1, 1, 1).cuda()/255,
                'bboxes': torch.from_numpy(boxes).repeat(count, 1).cuda(),
                'cls': torch.tensor(row['labels'], dtype=torch.float32).repeat(count)[:, None].cuda(),
                'batch_idx': torch.arange(count).repeat_interleave(len(boxes)).cuda()}
            model.zero_grad(set_to_none=True)
            before = len(samples)
            loss, _ = model(batch)
            loss.sum().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(loss).all()
            headroom = available_commit_bytes()
            results.append({'batch': count, 'commit_after': headroom,
                'minimum_sampled_commit': min((r['commit'] for r in samples[before:] if r['commit'] is not None), default=headroom),
                'reserved_gpu_mib': torch.cuda.memory_reserved()/2**20,
                'optimizer_steps': 0, 'accuracy_claim': False})
            print(json.dumps(results[-1]), flush=True)
            del batch, loss
    finally:
        stop.set()
        thread.join()
        Path('data/generations/yolo-g10c-guarded-1024/commit-probe.json').write_text(json.dumps({
            'scope': 'Capacity only; no optimizer steps or persisted weights; excludes full trainer/evaluation overhead',
            'samples': samples, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
