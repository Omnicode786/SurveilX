import gc,json,time,torch
from pathlib import Path
from training.detector_model import SVADetector,detection_loss
from training.detection_pipeline import DetectionImages,collate_detection
from training.datasets import digest
assert torch.cuda.is_available()
torch.set_num_threads(3); torch.cuda.set_per_process_memory_fraction(.75)
root=Path('data/datasets/sh17-development-v2'); manifest=json.loads((root/'manifest.json').read_text()); samples=[s for s in manifest['samples'] if s['split']=='train' and s['labels']]
dataset=DetectionImages(root,samples,512,augment=False)
results=[]
for batch in (1,2,4):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(42)
    model=SVADetector(17,architecture_version=3).cuda().train(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    images,context,zones,targets=collate_detection([dataset[i] for i in range(batch)])
    images,context,zones=[v.cuda() for v in (images,context,zones)]
    times=[]
    try:
        for step in range(3):
            torch.cuda.synchronize(); start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            loss=detection_loss(model(images,context,zones),targets,quality_warmup=1.)['loss']; assert torch.isfinite(loss)
            loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step(); torch.cuda.synchronize(); times.append(time.perf_counter()-start)
        entry={'batch':batch,'image_size':512,'seconds':times,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'finite_loss':float(loss.detach()),'scope':'temporary training-split profiling; weights never saved'}
    except torch.cuda.OutOfMemoryError as exc:
        entry={'batch':batch,'error':'CUDA OOM under 75 percent memory ceiling','detail':str(exc)}
    results.append(entry); print(json.dumps(entry),flush=True)
    del model,optimizer,images,context,zones
    if 'loss' in locals(): del loss
    gc.collect(); torch.cuda.empty_cache()
    if entry.get('error') or entry.get('peak_reserved_mib',0)>1100: break
Path('reports/sva-v3-cuda-profile-1013.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'dataset_sha256':digest(root/'manifest.json'),'memory_fraction':.75,'results':results},indent=2))
