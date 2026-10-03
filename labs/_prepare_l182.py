"""Authenticate inherited paper release, local inputs and frozen fresh operator."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l182'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ledger=json.loads((P/'sources/l166/source-ledger.json').read_text());protected={}
for n,h in ledger['files'].items():
 p=P/'sources/l166'/n
 assert digest(p)==h,n
 protected[str(p.relative_to(R))]=h
manifest=json.loads((P/'evidence/l166/input-manifest.json').read_text());root=Path('/tmp/l166-input')
assert digest(root/'input-manifest.json')==digest(P/'evidence/l166/input-manifest.json')
for n,v in manifest['files'].items():assert digest(root/n)==v['sha256'],n
x=np.load(root/'prepared.npz')
assert x['X_train'].shape[0]==11411 and x['X_test'].shape[0]==702
assert len(set(map(tuple,x['test_keys'])))==702
for s in range(10):
 seed=int.from_bytes(hashlib.sha256(f'rel-f1-dfs-2:driver-dnf:{s}'.encode()).digest()[:4],'big')
 expected=np.random.default_rng(seed).choice(11411,512,replace=False)
 np.testing.assert_array_equal(expected,x['support'][s])
 assert len(set(expected))==512
 assert not (set(map(tuple,x['train_keys'][expected])) & set(map(tuple,x['test_keys'])))
# L166 budget seals the actually executed runner, independent of today's operator.
old=json.loads((P/'evidence/l166/budget.json').read_text())
assert digest(P/'_run_l166.py')==old['source_hashes']['labs/_run_l166.py']
for n in ['labs/_run_l166.py','labs/_guard_l182.py','modal/l182_repro.py','labs/evidence/l166/input-manifest.json','labs/evidence/l166/prepared.npz','labs/l166-reproduction.md','labs/relkit/rdbpfn_l166.py']:
 protected[n]=digest(R/n)
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(P/'sources/l182/source-ledger.json').write_text(json.dumps(dict(papers=[ledger['paper'],'https://arxiv.org/html/2502.06784v2'],inherited_files=protected,scope='Fresh unchanged checkpoint evaluation; proposed hybrid NOT_RUN',full_dfs_regeneration='NOT_RUN',historical_identity='NOT_ESTABLISHED'),indent=2)+'\n')
budget=E/'budget.json'
if budget.exists():raise RuntimeError('Do not overwrite budget')
budget.write_text(json.dumps(dict(cap_usd=10,planned_stop_usd=8,overhead_reserve_usd=2,max_worker_seconds=6000,source_hashes=protected,reservations=[]),indent=2)+'\n')
r=dict(status='PASS',authenticated_source_files=len(protected),complete_queries=702,support_seeds=10,support_rows=512,runner='UNCHANGED_AUTHENTICATED_L166',label_orientation='RELEASED_COMPLEMENT_OF_CURRENT_RAW_DNF',feature_availability='NOT_ESTABLISHED')
(E/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
