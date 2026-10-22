"""Bounded CUDA capacity probe; local training samples, discarded weights, no accuracy claim."""
import gc,json,time,statistics
from pathlib import Path
import torch
from training.datasets import digest
from training.detection_pipeline import DetectionImages,collate_detection
from training.detector_model import SVADetector,detection_loss
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

torch.set_num_threads(3)
torch.cuda.set_per_process_memory_fraction(0.85)
report={'scope':'Local train-sample forward/backward/AdamW capacity only; no saved weights or accuracy evaluation',
        'gpu':torch.cuda.get_device_name(0),'memory_fraction':0.85,'cpu_threads':3,'measurements':[]}
output=Path('reports/g8-capacity-probe.json');assert not output.exists()

def measure(kind,version,dataset,size,batch):
    directory=Path('data/runs')/version
    metadata=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert digest(directory/'weights.pt')==metadata['weights_sha256']
    manifest_path=Path('data/datasets')/dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    samples=sorted([s for s in manifest['samples'] if s['split']=='train'],key=lambda s:len(s['labels']),reverse=True)
    data=DetectionImages(manifest_path.parent,samples,size,decode_mode='bounded-jpeg-v1')
    images,contexts,zones,targets=collate_detection([data[i] for i in range(batch)])
    images,contexts,zones=[v.cuda() for v in (images,contexts,zones)]
    if kind=='yolo':
        model=YOLO(str(directory/'weights.pt')).model.float().cuda().train()
        model.args=get_cfg(overrides=model.args if isinstance(model.args,dict) else vars(model.args))
        boxes=torch.cat([target['boxes'] for target in targets]).cuda()
        values={'img':images,'batch_idx':torch.cat([torch.full((len(t['boxes']),),i,device='cuda') for i,t in enumerate(targets)]),
            'cls':torch.cat([t['labels'] for t in targets]).float().reshape(-1,1).cuda(),
            'bboxes':torch.cat([(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]],-1)}
        loss=lambda:model(values)[0].sum()
    else:
        model=SVADetector(**metadata['model_config']).cuda().train()
        model.load_state_dict(torch.load(directory/'weights.pt',map_location='cuda',weights_only=True))
        loss=lambda:detection_loss(model(images,contexts,zones),targets)['loss']
    for p in model.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
    times=[]
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        value=loss();assert torch.isfinite(value);value.backward();optimizer.step();torch.cuda.synchronize()
        if step:times.append(time.perf_counter()-start)
    return {'status':'completed','batch':batch,'image_size':size,'model':version,'kind':kind,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'median_step_seconds':statistics.median(times),'images_per_second':batch/statistics.median(times),
        'objects_in_batch':sum(len(t['labels']) for t in targets),'saved_weights':False}

probes=[('yolo','cuda-g4-ppe-yolo','sh17-development-v2',512,b) for b in (4,8,12)]
probes += [('yolo','accuracy-g3-weapons-yolo','dangerous-items-development-v1',448,8),
    ('scratch','cuda-g6-weapons-sva-v3','dangerous-items-development-v1',384,4)]
for kind,version,dataset,size,batch in probes:
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    try:result=measure(kind,version,dataset,size,batch)
    except torch.cuda.OutOfMemoryError:
        result={'status':'oom_at_budget','batch':batch,'image_size':size,'model':version,'kind':kind}
    report['measurements'].append(result);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    gc.collect();torch.cuda.empty_cache()
report['state']='completed';output.write_text(json.dumps(report,indent=2),encoding='utf-8')
