import json,copy,time,shutil
from pathlib import Path
from scripts.train_accuracy_campaign import verify_job,command_for
from training.datasets import digest
source=Path('data/generations/accuracy-g8b-corrected-1015')
plan=json.loads((source/'plan.json').read_text(encoding='utf-8'))
name='accuracy-g8c-capacity-1015';directory=Path('data/generations')/name
assert not directory.exists()
limits={'weapons-yolo':(24,8),'ppe-yolo':(24,8),'ppe-sva-v3':(24,8),'weapons-sva-v3':(24,8),
 'fire-yolo':(24,8),'fire-sva':(16,6),'fighting':(12,4),'fall':(4,2),'person':(6,2)}
checks=[]
for job in plan['jobs']:
    job['version']=job['version'].replace('corrected-g8b-','capacity-g8c-')
    assert not (Path('data/runs')/job['version']).exists()
    suffix=job['version'].removeprefix('capacity-g8c-');epochs,patience=limits[suffix]
    job['options'].update(threads=3,epochs=epochs,patience=patience)
    if job['pipeline']=='yolo':job['options']['batch']=8
    if suffix=='ppe-sva-v3':job['options'].update({'batch-size':2,'accumulation-steps':2})
    job['rationale']+=' User requested greater GPU throughput/epoch headroom. Batch8 YOLO and batch2 PPEv3 were measured on this GPU; larger failing batches rejected; ceilings retain early stopping.'
    verify_job(job)
    checks.append({'version':job['version'],'command':command_for(job)})
    print('Verified',job['version'],flush=True)
plan.update(name=name,created=time.time(),gpu_memory_fraction=0.85,minimum_free_gpu_mib=1850,
    hardware='MX130/2GiB FP32; measured YOLO batch8, PPEv3 batch2/accum2, weaponsv3 batch2/accum2; three CPU threads; serial.85% allocator budget is not compute limit.',
    capacity_evidence={'reports/g8-capacity-probe.json':digest(Path('reports/g8-capacity-probe.json')),
        'reports/g8-ppe-capacity-probe.json':digest(Path('reports/g8-ppe-capacity-probe.json'))},
    supersedes='Original G8 failed dict-loss callback and was stopped; G8b prepared but never launched. New outputs preserve both plans/checkpoints; recorder supports both loss formats.')
directory.mkdir(parents=True,exist_ok=False)
for relative,fingerprint in plan['jobs'][0]['code_sha256'].items():
    path=Path(relative);assert digest(path)==fingerprint
    target=directory/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
(directory/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(directory/'preflight.json').write_text(json.dumps({'jobs':checks,'plan_sha256':digest(directory/'plan.json'),
    'capacity_evidence':plan['capacity_evidence'],'note':'All existing manifests/assets/parents/source hashes reverified. No training launched by preparation.'},indent=2),encoding='utf-8')
print('FROZEN',digest(directory/'plan.json'))
