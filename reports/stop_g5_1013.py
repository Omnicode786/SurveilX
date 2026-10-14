import json,time,psutil,hashlib
from pathlib import Path
root=Path.cwd(); campaign=root/'data/generations/cuda-accuracy-g5-1013'; status=json.loads((campaign/'status.json').read_text()); runner=psutil.Process(status['pid'])
assert runner.cmdline()[1:4]==['-u','-m','scripts.train_accuracy_campaign'] and str(campaign/'plan.json') in runner.cmdline()
children=runner.children(recursive=True); snapshot=[{'pid':p.pid,'cmd':p.cmdline()} for p in [runner,*children]]
(campaign/'status-before-user-stop.json').write_text(json.dumps(status,indent=2))
runner.kill(); runner.wait(timeout=15)
for p in reversed(children):
    try: p.kill()
    except psutil.NoSuchProcess: pass
psutil.wait_procs(children,timeout=15)
status['state']='stopped_by_user'; status['stopped']=time.time(); status['stop_reason']='User requested architecture review and replacement campaign; completed and partial artifacts preserved'
for item in status['jobs'].values():
    if item.get('state')=='running': item['state']='interrupted'; item['stop_reason']=status['stop_reason']
(campaign/'status.json').write_text(json.dumps(status,indent=2))
lock=campaign/'runner.lock'
if lock.exists(): lock.rename(campaign/'runner.stopped-by-user.lock')
report={'stopped_processes':snapshot,'state':status['state'],'completed':[k for k,v in status['jobs'].items() if v.get('state')=='completed'],'interrupted':[k for k,v in status['jobs'].items() if v.get('state')=='interrupted'],'plan_sha256':hashlib.sha256((campaign/'plan.json').read_bytes()).hexdigest(),'preserved_partial_files':[str(p.relative_to(root)) for p in (root/'data/runs/cuda-g5-ppe-yolo').rglob('*') if p.is_file()]}
(root/'reports/g5-user-stop-1013.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
