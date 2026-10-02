"""Authenticate original evidence; freeze a declared nested-support intervention."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from relkit.few_shot_l176 import nested_support
P=Path(__file__).resolve().parent;E=P/'evidence/l176';S=P/'sources/l176';OUT=Path('/tmp/l176-input');OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(path,data):
    if path.exists():assert path.read_bytes()==data,'Frozen input changed: '+str(path)
    else:path.write_bytes(data)
old=json.loads((P/'evidence/l169/input-manifest.json').read_text());pins=json.loads((P/'evidence/l169/audit-manifest.json').read_text())
for name,h in pins['files'].items():assert sha(P/name)==h,name
for name,item in old['files'].items():assert sha(Path('/tmp/l169-input')/name)==item['sha256'],name
manifest=json.loads(json.dumps(old));manifest.update(experiment='L176 Nested-Support ICL Evaluation',fresh_contexts=old['contexts'],sampling='One original 1024 draw per task/seed; ordered prefixes for all k',reused_evaluations=0)
checks=[]
for spec in old['experiments']:
    db=spec['database'];data=np.load(P/'evidence/l169'/(db+'.npz'));arrays={k:data[k] for k in data.files if not k.startswith('support_')}
    schedules=[nested_support(len(data['y_train']),old['contexts'],spec['seed_key'].format(seed=s)) for s in range(10)]
    for k in old['contexts']:arrays['support_'+str(k)]=np.array([s[k] for s in schedules])
    assert np.array_equal(arrays['support_1024'],data['support_1024'])
    train=data['train_keys'];query=data['test_keys'];horizon=(60 if db=='rel-f1' else 365)*86400*10**9
    assert len(set(map(tuple,train)))==len(train) and len(set(map(tuple,query)))==len(query)
    assert not set(map(tuple,train))&set(map(tuple,query))
    assert (train[:,1]+horizon<query[:,1].min()).all()
    assert all(len(np.unique(data['y_train'][s[k]]))==2 for s in schedules for k in old['contexts'])
    dest=OUT/(db+'.npz');np.savez_compressed(dest,**arrays)
    freeze(E/dest.name,dest.read_bytes())
    manifest['files'][dest.name].update(sha256=sha(dest),bytes=dest.stat().st_size)
    checks.append(dict(database=db,train_rows=len(train),test_rows=len(query),features=spec['features'],horizon_days=60 if db=='rel-f1' else 365,nested_schedules=10,classes_present_at_every_k=True,query_support_overlap=0,all_label_horizons_completed=True))
for name in old['files']:
    if not name.endswith('.npz'):shutil.copyfile(Path('/tmp/l169-input')/name,OUT/name)
text=json.dumps(manifest,indent=2)+'\n';freeze(E/'input-manifest.json',text.encode());(OUT/'input-manifest.json').write_text(text)
freeze(E/'inherited-manifest.json',(P/'evidence/l169/audit-manifest.json').read_bytes())
freeze(E/'preflight.json',(json.dumps(dict(status='PASS',checks=checks,inherited_files_authenticated=len(pins['files']),original_dfs_regeneration='NOT_RUN',historical_availability='NOT_ESTABLISHED',label_orientation='Released complemented labels retained'),indent=2)+'\n').encode())
for name,path in [('rdbpfn.html',P/'sources/l169/rdbpfn.html'),('paper-targets.json',P/'sources/l169/paper-targets.json')]:freeze(S/name,path.read_bytes())
ledger=dict(paper='https://arxiv.org/html/2603.03805v5',code_revision=old['code_revision'],data_revision=old['data_revision'],tabicl_revision=old['tabicl_revision'],files={n:sha(S/n) for n in ['rdbpfn.html','paper-targets.json']},upstream_files=old['source_files'])
freeze(S/'source-ledger.json',(json.dumps(ledger,indent=2)+'\n').encode())
print(json.dumps(checks,indent=2))
