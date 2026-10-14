import copy,json,time,shutil
from pathlib import Path
from scripts.train_accuracy_campaign import verify_job,command_for
from training.datasets import digest
root=Path.cwd(); old=json.loads((root/'data/generations/cuda-accuracy-g5-1013/plan.json').read_text()); prepared=json.loads((root/'data/generations/sva-v3-prepared-1013/plan.json').read_text())
name='cuda-accuracy-g6-v3-1013'; directory=root/'data/generations'/name; directory.mkdir(exist_ok=True); assert not (directory/"plan.json").exists()
sources=[*Path('training').glob('*.py'),Path('scripts/train_accuracy_campaign.py'),Path('surveilx/accelerators.py'),Path('surveilx/training_runtime.py')]
code={p.as_posix():digest(p) for p in sources}
for p in sources:
    target=directory/'source_snapshot'/p; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
jobs=[]
for job in prepared['jobs']:
    job=copy.deepcopy(job); family=job['version'].split('-')[2]; job['version']=f'cuda-g6-{family}-sva-v3'; job['options'].update(epochs=48,patience=16); job['options']['batch-size']=1 if family=='ppe' else 2; jobs.append(job)
person=copy.deepcopy(next(j for j in old['jobs'] if j['version']=='cuda-g5-person-scratch')); person['version']='cuda-g6-person-sva-v3'; person['options']={'architecture-version':3,'transfer-from-legacy':True,'assignment':'coverage-v3','preprocessing':'stretch','epochs':32,'patience':12,'threads':3,'seed':42,'image-size':256,'batch-size':2,'learning-rate':.0003,'balanced-sampling':False}; person['rationale']='Same v3 detector and frozen train/holdouts; person-only infrastructure evidence'; jobs.append(person)
for original in old['jobs']:
    if original['pipeline']=='yolo':
        job=copy.deepcopy(original); job['version']=job['version'].replace('cuda-g5','cuda-g6'); job['options'].update(epochs=32,patience=12); job['rationale']='Continue adapted domain-conditioned YOLO, validation-only selection; no unsupported architecture replacement'
        if 'fire-yolo' in job['version']:
            job['parent']='cuda-g5-fire-yolo'; job['parent_sha256']=digest(Path('data/runs')/job['parent']/'weights.pt')
        jobs.append(job)
    elif original['pipeline']=='event':
        job=copy.deepcopy(original); job['version']=job['version'].replace('cuda-g5','cuda-g6'); jobs.append(job)
v2=copy.deepcopy(jobs[-1]); v2.update(version='cuda-g6-entity-v2-event',parent='accuracy-g2-entity-v2',dataset='generated-v2',dataset_sha256=digest(Path('data/datasets/generated-v2/manifest.json'))); v2['parent_sha256']=digest(Path('data/runs')/v2['parent']/'weights.pt'); v2['options'].update(epochs=8,patience=4); v2['rationale']='Bounded synthetic v2 retention check only; saturated synthetic scores are not real-domain accuracy'; jobs.append(v2)
checks=[]
for job in jobs:
    job['code_sha256']=code
    manifest,parent,counts=verify_job(job)
    checks.append({'version':job['version'],'counts':counts,'command':command_for(job),'parent_sha256':job['parent_sha256']}); print('Verified '+job['version'],flush=True)
plan={**old,'name':name,'created':time.time(),'jobs':jobs,'gpu_policy':'Serial CUDA float32, v3 batch1 at512 and batch2 at384/256; measured reserved820/1000MiB; YOLO8/event8 previously measured','unsupported':['No downloads, acceptance or activation. Theft, collision and learned state/relation supervision remains missing.','Entity v1/v2 are synthetic retention checks; person baseline does not establish multi-domain capability.','V3 is a bundled architecture experiment; no gain claimed before validation.','Interrupted g5 PPE YOLO has no completed checkpoint; preserved partial files, new version initialized from completed g4 parent.']}
(directory/'plan.json').write_text(json.dumps(plan,indent=2)); (directory/'preflight.json').write_text(json.dumps({'checks':checks,'code_sha256':code,'plan_sha256':digest(directory/'plan.json'),'disk_free':shutil.disk_usage(root).free},indent=2)); print(json.dumps({'jobs':len(jobs),'plan_sha256':digest(directory/'plan.json'),'directory':str(directory)},indent=2))

