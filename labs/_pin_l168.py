"""Freeze author evidence once; refuse silent changes to an existing audit packet."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;E=P/'evidence/l168'
experiments=[dict(database='rel-f1',task='driver-dnf',folder='evidence/l166',train_rows=11411,test_rows=702,features=72,
 seed_key='rel-f1-dfs-2:driver-dnf:{seed}',phases=['pilot-2','full-1'],evidence='REUSED_L166',targets={'RDBPFN':.7219,'RDBPFN_single':.6640,'TabICLv1.1':.7176}),
 dict(database='rel-trial',task='study-outcome',folder='evidence/l168',train_rows=11994,test_rows=825,features=176,
 seed_key='rel-trial-dfs-2:study-outcome:{seed}',phases=['pilot-1','full-1'],evidence='FRESH_L168',targets={'RDBPFN':.5986,'RDBPFN_single':.5961,'TabICLv1.1':.5926})]
files={}
for spec in experiments:
 folder=P/spec['folder']
 for p in [folder/'input-manifest.json',folder/'prepared.npz']+[p for phase in spec['phases'] for p in (folder/phase).iterdir() if p.suffix in ['.json','.npz']]:
  files[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((E/'input-manifest.json').read_text())
for name in list(manifest['source_files'])+['sources/l166/source-ledger.json','sources/l168/study-outcome.zip','sources/l168/metadata.yaml']:
 files[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
packet=dict(experiments=experiments,files=dict(sorted(files.items())))
p=E/'audit-manifest.json'
if p.exists():assert json.loads(p.read_text())==packet,'Immutable evidence changed'
else:p.write_text(json.dumps(packet,indent=2)+'\n')
b=json.loads((E/'budget.json').read_text());seconds=sum(json.loads((E/phase/'cost.json').read_text())['worker_body_seconds'] for phase in ['pilot-1','full-1'])
cost=dict(cap_usd=10,overhead_reserved_usd=3,total_reserved_usd=3+sum(r['upper_usd'] for r in b['reservations']),known_worker_body_seconds=seconds,
 known_worker_body_usd=seconds*.00028372,invoice='NOT_ITEMIZED',all_seeds_complete=True,retries=0,
 setup_failure='One local entrypoint failed before worker dispatch; log retained; preparation covered by overhead reserve',
 collection_note='First download destination was not a directory; recollected into a directory, verified all hashes; no compute rerun',
 apps=['ap-rlhyXY13XU3onpCwJm1aEw','ap-pSdJlBUBLezTBcxsn6VEKZ','ap-4NzS2mKLkyiCHKXRE1H9s0'])
(E/'cost.json').write_text(json.dumps(cost,indent=2)+'\n');print(cost)
