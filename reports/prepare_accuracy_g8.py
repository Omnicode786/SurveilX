import json,time,shutil
from pathlib import Path
from scripts.train_accuracy_campaign import verify_job,command_for
from training.datasets import digest
from surveilx.training_runtime import available_commit_bytes
import psutil
name='accuracy-g8-corrected-1015'
directory=Path('data/generations')/name
assert not directory.exists(), 'Preserve existing campaign; do not overwrite it'
jobs=[]
common={'threads':2,'seed':42}
def add(version,parent,dataset,pipeline,options,reason):
    assert not (Path('data/runs')/version).exists()
    jobs.append({'version':version,'parent':parent,'dataset':dataset,'pipeline':pipeline,
        'options':{**common,**options},'rationale':reason,
        'dataset_sha256':digest(Path('data/datasets')/dataset/'manifest.json'),
        'parent_sha256':digest(Path('data/runs')/parent/'weights.pt')})
def yolo(version,parent,dataset,size,epochs,patience,rate,geometry='stretch',baseline=False):
    options={'epochs':epochs,'patience':patience,'imgsz':size,'batch':4,'learning-rate':rate,
        'preprocessing':geometry,'training-profile':'finetune-v2','effective-batch-size':8}
    if baseline:options['baseline']=True
    add(version,parent,dataset,'yolo',options,
        'Corrected zero bias warmup, bounded schedule/effective batch, lossless data, mild augmentation and native validation AP50 selection. New version; validation only selection; no deployment.')
yolo('corrected-g8-weapons-yolo','accuracy-g3-weapons-yolo','dangerous-items-development-v1',448,12,6,0.0001,'letterbox')
yolo('corrected-g8-ppe-yolo','cuda-g4-ppe-yolo','sh17-development-v2',512,16,6,0.0001)
partial=Path('data/runs/routing-g7b-ppe-sva-v3')
add('corrected-g8-ppe-sva-v3','cuda-g4-ppe-scratch','sh17-development-v2','scratch',
    {'epochs':16,'patience':6,'image-size':512,'batch-size':1,'learning-rate':0.0001,
     'architecture-version':3,'transfer-from-legacy':True,'assignment':'coverage-v3','preprocessing':'stretch',
     'balanced-sampling':True,'accumulation-steps':4,'augmentation':'robust-v2','training-decode-mode':'bounded-jpeg-v1',
     'initialize-partial':str(partial),'partial-sha256':digest(partial/'weights.pt'),
     'partial-dataset-sha256':digest(Path('data/datasets/sh17-development-v2/manifest.json'))},
    'Preserved best g7b partial weights in NEW output; not optimizer resume. Bounded JPEG pixel buffers and rich class/IoU/decode telemetry; no quality-bootstrap reset.')
add('corrected-g8-weapons-sva-v3','cuda-g6-weapons-sva-v3','dangerous-items-development-v1','scratch',
    {'epochs':12,'patience':6,'image-size':384,'batch-size':2,'learning-rate':0.00005,
     'architecture-version':3,'assignment':'coverage-v3','preprocessing':'stretch','balanced-sampling':True,
     'accumulation-steps':2,'augmentation':'robust-v2','training-decode-mode':'bounded-jpeg-v1'},
    'Bounded low-LR full-network continuation from completed v3; no new heads/bootstrap. Classification/confidence telemetry must show whether its weak learning improves; no guaranteed gain.')
yolo('corrected-g8-fire-yolo','cuda-g5-fire-yolo','dfire-development-v2',384,12,5,0.000075)
add('corrected-g8-fire-sva','cuda-g4-fire-scratch','dfire-development-v2','scratch',
    {'epochs':8,'patience':4,'image-size':384,'batch-size':2,'learning-rate':0.00005,
     'balanced-sampling':True,'classification-focal-gamma':2.0,'classification-focal-alpha':0.25,
     'accumulation-steps':2,'augmentation':'robust-v2','training-decode-mode':'bounded-jpeg-v1'},
    'Preserve successful legacy fire head/focal objective, conservative LR and bounded decode. Retain parent on aggregate regression.')
add('corrected-g8-fighting','routing-g7-fighting','airtlab-fighting-development-v1','event',
    {'epochs':6,'patience':3,'batch-size':8,'learning-rate':0.00005,'augment':True,'finetune-all':True},
    'Bounded continuation from validation-selected g7 event model; limited staged clips cannot establish independent-site reliability.')
add('corrected-g8-fall','accuracy-g2-fall','urfall-development-v1','event',
    {'epochs':2,'patience':2,'batch-size':8,'learning-rate':0.000025,'augment':True,'finetune-all':True},
    'Short retention/stress continuation only; perfect tiny test has no measured accuracy headroom.')
yolo('corrected-g8-person','accuracy-g2-person-baseline','pennfudan',256,4,2,0.00005,baseline=True)
sources=[*sorted(Path('training').glob('*.py')),Path('scripts/train_accuracy_campaign.py'),
    Path('surveilx/accelerators.py'),Path('surveilx/training_runtime.py'),Path('surveilx/detector_expert.py'),Path('surveilx/model_profiles.py')]
code={p.as_posix():digest(p) for p in sources}
checks=[]
for job in jobs:
    job['code_sha256']=code
    manifest,parent,counts=verify_job(job)
    checks.append({'version':job['version'],'counts':counts,'command':command_for(job)})
    print('Verified',job['version'],flush=True)
plan={'name':name,'created':time.time(),'jobs':jobs,'target':0.5,'required_cuda':True,'gpu_memory_fraction':0.75,
 'minimum_free_disk_bytes':1800000000,'minimum_available_ram_bytes':3500000000,
 'minimum_available_commit_bytes':3500000000,'minimum_free_gpu_mib':1400,
 'selection':'Validation only; calibration separate; test reports only. No acceptance, deployment or automatic promotion.',
 'hardware':'MX130 sm50/2GiB; FP32; serial jobs, two CPU threads, measured conservative batches; allocator headroom is not a compute cap.',
 'unsupported':['No new datasets/downloads; original licensing and site-independence limitations remain.',
 'Theft, collision, traffic violations and industrial state/relation tasks lack suitable existing annotated supervision; no fabricated training.',
 'Sparse PPE classes cannot be repaired by architecture alone; repeated development tests are not independent acceptance.',
 'No perfect-accuracy guarantee; changes require empirical validation. Person/fall runs are short retention checks.',
 'No redundant synthetic or identical inherited-model copies are retrained. Full hardware/production deployment and research paper remain excluded.']}
directory.mkdir(parents=True,exist_ok=False)
for path in sources:
    target=directory/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
(directory/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(directory/'preflight.json').write_text(json.dumps({'jobs':checks,'code_sha256':code,'plan_sha256':digest(directory/'plan.json'),
    'free_disk_bytes':shutil.disk_usage('.').free,'available_ram_bytes':psutil.virtual_memory().available,
    'available_commit_bytes':available_commit_bytes(),'training_not_launched_by_preparation':True},indent=2),encoding='utf-8')
print('Frozen plan',digest(directory/'plan.json'),'jobs',len(jobs),'sources',len(sources))
