"""Freeze complete L189 and L197 packets; preserve all original file identities."""
import hashlib,json,platform,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l198';Q=E/'packet';Q.mkdir(parents=True,exist_ok=True)
files={};origins={}
def put(source,name):
    data=source.read_bytes();h=hashlib.sha256(data).hexdigest();dest=Q/name
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    files[name]=h;origins[name]={'path':str(source.relative_to(P.parent)),'sha256':h}
for lesson in ['l189','l197']:
    origin=P/'evidence'/lesson;m=json.loads((origin/'input-manifest.json').read_text())
    for name,h in m['files'].items():
        source=origin/'packet'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==h,name
        put(source,f'evidence/{lesson}/packet/{name}')
    for name in ['input-manifest.json','report.json']:
        put(origin/name,f'evidence/{lesson}/{name}')
    for prefix in ['_audit_','_verify_','_test_']:
        put(P/(prefix+lesson+'.py'),prefix+lesson+'.py')
for name in ['gaps_l189','landscape_l197']:
    put(P/'relkit'/(name+'.py'),'relkit/'+name+'.py')
(E/'input-manifest.json').write_text(json.dumps({'files':files,'origins':origins},indent=2)+'\n')
protocol=dict(experiment='L198-RESEARCH-PROPOSAL-AUDIT',candidate_ids=['temporal','composite','transfer'],weight_grid=[1,2,3],weight_scenarios=27,predecessor_reports=['L189','L197'],table_task_cells=401,table_summary_cells=90,prediction_rows=33650,l149_seeds=list(range(5)),l182_draws=list(range(10)),l182_arms=3,bootstrap_draws=2000,bootstrap_seed=137,rdblearn_tasks=21,cloud_usd=0,cap_seconds=1800,future_models='NOT_RUN',novelty='NOT_ESTABLISHED',scope='Full original ranking and L197 evidence replay; new proposal matrices and illustrative contrasts; no new training or inference')
(E/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(E/'environment.txt').write_text(platform.python_version()+'\n'+subprocess.check_output([str(P.parent/'.venv/bin/python'),'-m','pip','freeze'],text=True))
print('Frozen',len(files),'original files')
