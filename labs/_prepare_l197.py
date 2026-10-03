"""Freeze full predecessor packets and exact executable sources without changing them."""
import hashlib,json,shutil,platform,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l197';Q=E/'packet';Q.mkdir(parents=True,exist_ok=True)
files={};origins={}
def put(src,relative):
    dest=Q/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    data=src.read_bytes();digest=hashlib.sha256(data).hexdigest();dest.write_bytes(data)
    files[relative]=digest;origins[relative]=dict(path=str(src.relative_to(P.parent)),sha256=digest)
for lesson in ['l191','l195']:
    origin=P/'evidence'/lesson
    manifest=json.loads((origin/'input-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        source=origin/'packet'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
        put(source,f'evidence/{lesson}/packet/{name}')
    for name in ['input-manifest.json','report.json']:
        put(origin/name,f'evidence/{lesson}/{name}')
    for prefix in ['_replay_','_verify_']:
        put(P/(prefix+lesson+'.py'),prefix+lesson+'.py')
for name in ['tracking_l191','stress_l195']:
    put(P/'relkit'/(name+'.py'),'relkit/'+name+'.py')
(E/'input-manifest.json').write_text(json.dumps(dict(files=files,origins=origins),indent=2)+'\n')
protocol=dict(experiment='L197-LANDSCAPE-EVIDENCE-AUDIT',tables=[3,4,7,8],table_task_cells=401,table_summary_cells=90,prediction_rows=33650,l149_seeds=list(range(5)),l182_support_draws=list(range(10)),l182_arms=3,rdblearn_tasks=21,bootstrap_draws=2000,bootstrap_seed=137,cloud_usd=0,cap_seconds=1800,mode='Complete saved-evidence reconstruction and replay; no new model execution',source_versions=['KumoRFM-2 2604.12596v1','RDB-PFN 2603.03805v5','RDBLearn 2602.18495v1'],source_freeze='Inherited authenticated L191/L195 packets; current web readings do not replace their bytes')
(E/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(E/'environment.txt').write_text(platform.python_version()+'\n'+subprocess.check_output([str(P.parent/'.venv/bin/python'),'-m','pip','freeze'],text=True))
print('Frozen',len(files),'files')
