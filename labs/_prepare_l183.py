"""Pin inherited evidence against Git HEAD and existing source ledgers; archive primary HTML."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import requests
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l183';Q=E/'packet';S=P/'sources/l183'
Q.mkdir(parents=True,exist_ok=True);S.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=['labs/evidence/l146/summary.json','labs/evidence/l146/paper-table.json','labs/evidence/l146/sources.json','labs/evidence/l145/prepared/audit.json','labs/evidence/l145/cost-decision.json','labs/evidence/l164/cost-decision.json','labs/evidence/l164/pilot.json','labs/evidence/l164/source-audit.json','labs/evidence/l164/delivery-manifest.json','labs/l145-reproduction.md','labs/l146-reproduction.md','labs/l164-reproduction.md','labs/relkit/relgt_l145.py','labs/_full_l145.py','labs/relkit/griffin_l164.py','labs/_run_l164.py','labs/sources/l164/upstream/hmodel.py','labs/sources/l164/upstream/hmaintask_downsample_absolute_eval_sample.py','labs/sources/l164/upstream/LICENSE','labs/sources/l145/LICENSE','labs/sources/l165/paper.txt','labs/sources/l165/source-ledger.json']
for arm in ['gnn','relgt']:
 for seed in range(3):
  paths += [f'labs/evidence/l146/fit-{arm}-{seed}/{name}' for name in ['predictions.npz','result.json']]
paths += ['labs/sources/l164/upstream/task_names.yaml','labs/evidence/l164/budget.json','labs/evidence/l145/pilot-1/result.json']
paths += [f'labs/evidence/l146/prepared/{s}.npz' for s in ['val','test']]
files={}
for name in paths:
 raw=(R/name).read_bytes();historical=subprocess.check_output(['git','show','HEAD:'+name],cwd=R)
 assert raw==historical,'Inherited file differs from HEAD: '+name
 dest=Q/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);files[name]=sha(raw)
old=json.loads((P/'evidence/l146/sources.json').read_text())['reused_files']
for name in ['relkit/relgt_l145.py','_full_l145.py']:assert files['labs/'+name]==old[name]
old=json.loads((P/'evidence/l164/delivery-manifest.json').read_text())['files']
for name in ['labs/relkit/griffin_l164.py','labs/_run_l164.py']:assert files[name]==old[name]
ledger=[]
for name,url in [('relgt.html','https://arxiv.org/html/2505.10960v1'),('griffin.html','https://arxiv.org/html/2505.05568v1'),('frontier.html','https://docs.nvidia.com/sdgm/research/relational-graph-transformers')]:
 path=S/name
 if not path.exists():
  response=requests.get(url,timeout=50);response.raise_for_status();path.write_bytes(response.content)
 ledger.append(dict(file=name,url=url,sha256=sha(path.read_bytes()),bytes=path.stat().st_size,retrieved='2026-10-02'))
manifest=dict(files=files,inherited_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),authentication='Byte equality with Git HEAD; canonical model/trainer hashes also match original lesson ledgers',boundary='Saved evidence only; raw database labels, full token caches and model weights are not rerun or reauthenticated here')
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(S/'source-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print('Pinned',len(files),'inherited files and',len(ledger),'primary pages')
