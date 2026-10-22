import json,time,shutil
from pathlib import Path
from scripts.train_accuracy_campaign import verify_job,command_for
from training.datasets import digest
root=Path.cwd(); name='accuracy-routing-g7-1015'; directory=Path('data/generations')/name; directory.mkdir(exist_ok=False)
old=json.loads(Path('data/generations/cuda-accuracy-g6-v3-1013/plan.json').read_text())
by={j['version']:j for j in old['jobs']}
jobs=[]
def add(version,parent,dataset,pipeline,options,rationale):
    jobs.append({'version':version,'parent':parent,'dataset':dataset,'pipeline':pipeline,'options':options,'rationale':rationale,
        'dataset_sha256':digest(Path('data/datasets')/dataset/'manifest.json'),'parent_sha256':digest(Path('data/runs')/parent/'weights.pt')})
common={'threads':2,'seed':42}
add('routing-g7-weapons-letterbox','accuracy-g3-weapons-yolo','dangerous-items-development-v1','yolo',
    {**common,'epochs':12,'patience':6,'imgsz':448,'batch':4,'learning-rate':.0001,'preprocessing':'letterbox'},
    'Validation-only class routing favors g3 for firearm/machete/bat recall; test aspect-preserving geometry, preserve g4 knife comparator. New calibration required; no activation.')
add('routing-g7-fighting','cuda-fighting-g2-scene','airtlab-fighting-development-v1','event',
    {**common,'epochs':8,'patience':4,'batch-size':8,'learning-rate':.0001,'augment':True,'finetune-all':True},
    'Real development scene clips, validation-selected parent; bounded full-network photometric continuation, weak scene labels remain a limitation.')
partial=Path('data/runs/cuda-g6-ppe-sva-v3')
add('routing-g7-ppe-sva-v3','cuda-g4-ppe-scratch','sh17-development-v2','scratch',
    {**common,'epochs':8,'patience':4,'image-size':512,'batch-size':1,'learning-rate':.0001,
     'architecture-version':3,'transfer-from-legacy':True,'assignment':'coverage-v3','preprocessing':'stretch',
     'balanced-sampling':True,'accumulation-steps':4,'augmentation':'robust-v2',
     'initialize-partial':str(partial),'partial-sha256':digest(partial/'weights.pt'),
     'partial-dataset-sha256':by['cuda-g6-ppe-sva-v3']['dataset_sha256']},
    'New version from preserved interrupted g6 weights, not optimizer resume. Parent lineage/assets/hash checked. Effective batch4; no quality-bootstrap reset, no upscaling/shrinking tiny PPE labels.')
add('routing-g7-ppe-yolo','cuda-g4-ppe-yolo','sh17-development-v2','yolo',
    {**common,'epochs':12,'patience':6,'imgsz':512,'batch':4,'learning-rate':.0001,'preprocessing':'stretch'},
    'Strongest modern SH17 baseline for helmet/gloves/vest/mask; keep tested resolution and bounded CUDA batch.')
add('routing-g7-fire-yolo','cuda-g5-fire-yolo','dfire-development-v2','yolo',
    {**common,'epochs':8,'patience':4,'imgsz':384,'batch':4,'learning-rate':.000075,'preprocessing':'stretch'},
    'Validation-selected smoke baseline; g4 flame comparator retained; fixed splits, no test-based selection.')
add('routing-g7-fire-scratch','cuda-g4-fire-scratch','dfire-development-v2','scratch',
    {**common,'epochs':8,'patience':4,'image-size':384,'batch-size':2,'learning-rate':.0001,
     'balanced-sampling':True,'classification-focal-gamma':2.,'classification-focal-alpha':.25,'accumulation-steps':2,'augmentation':'robust-v2'},
    'Continue successful frozen fire architecture rather than resetting its head; effective batch4, mild photometric stress training.')
add('routing-g7-fall','accuracy-g2-fall','urfall-development-v1','event',
    {**common,'epochs':4,'patience':2,'batch-size':8,'learning-rate':.00005,'augment':True,'finetune-all':True},
    'Bounded real-development retention check; five positive validation and test clips do not establish general fall reliability.')
add('routing-g7-person','accuracy-g2-person-baseline','pennfudan','yolo',
    {**common,'epochs':4,'patience':2,'imgsz':256,'batch':4,'learning-rate':.00005,'baseline':True,'preprocessing':'stretch'},
    'Preserve validation-selected strong person baseline; short retention trial, not evidence of multi-domain fulfillment.')
sources=[*Path('training').glob('*.py'),Path('scripts/train_accuracy_campaign.py'),Path('surveilx/accelerators.py'),Path('surveilx/training_runtime.py'),Path('surveilx/detector_expert.py'),Path('surveilx/model_profiles.py')]
code={p.as_posix():digest(p) for p in sources}
checks=[]
for job in jobs:
 job['code_sha256']=code
 manifest,parent,counts=verify_job(job)
 checks.append({'version':job['version'],'counts':counts,'command':command_for(job)})
 print('Verified '+job['version'],flush=True)
plan={'name':name,'created':time.time(),'jobs':jobs,'target':.5,'selection':'Validation only; calibration separate; test reporting; no acceptance or activation',
 'required_cuda':True,'gpu_memory_fraction':.75,'minimum_free_disk_bytes':1500000000,'minimum_available_ram_bytes':1200000000,'minimum_free_gpu_mib':1400,
 'hardware':'NVIDIA MX130 sm50/2GiB; float32, serial jobs, batch4 YOLO, batch1 PPE SVA, effective4 by accumulation',
 'unsupported':['No new dataset downloads; real data licenses remain restrictive for PPE, fall and fighting.','Theft/collision/state/relation labels and independent deployment-site labels unavailable.','Saturated synthetic models and repeated weak weapons scratch training omitted: no product-accuracy benefit justified.','Retained weights with a changed geometry form a new calibrated variant, not a claim of retaining original accuracy.']}
for p in sources:
 target=directory/'source_snapshot'/p; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
(directory/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(directory/'preflight.json').write_text(json.dumps({'jobs':checks,'code_sha256':code,'plan_sha256':digest(directory/'plan.json'),'disk_free':shutil.disk_usage('.').free},indent=2),encoding='utf-8')
print('PLAN',digest(directory/'plan.json'))
