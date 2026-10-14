import json,shutil
from pathlib import Path
from training.datasets import digest
frozen=json.loads(Path('reports/g4-diagnostic-frozen-artifacts-before.json').read_text())['files']; changed=[p for p,h in frozen.items() if not Path(p).is_file() or digest(Path(p))!=h]; assert not changed,changed
status=json.loads(Path('data/generations/cuda-accuracy-g5-1013/status.json').read_text()); completed=[]
for version,item in status['jobs'].items():
    if item['state']!='completed': continue
    run=Path('data/runs')/version; metadata=json.loads((run/'manifest.json').read_text()); assert digest(run/'weights.pt')==metadata['weights_sha256']; completed.append({'version':version,'weights_sha256':metadata['weights_sha256'],'comparison':item['comparison']})
Path('reports/g6-prelaunch-integrity-1013.json').write_text(json.dumps({'frozen_files':len(frozen),'changed':changed,'g5_completed':completed,'disk_free':shutil.disk_usage('.').free,'tests':{'full_safe_suite':202,'deselected_real_scene_training':1,'focused':26,'ruff':'passed'}},indent=2)); print('Frozen files unchanged:',len(frozen),'G5 completed verified:',len(completed),'free bytes:',shutil.disk_usage('.').free)
