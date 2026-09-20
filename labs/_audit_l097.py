"""Independent metric, candidate, initialization and coverage audit; no scorer imports."""
import hashlib,io,json,math,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;r=json.loads((P/'_experiment_l097_results.json').read_text())
assert {(x['fold'],x['seed'],x['arm']) for x in r['runs']}=={(f,s,a) for f in range(1,6) for s in range(3) for a in ['uniform','degree','hard']}
assert r['source_sha256']==hashlib.sha256((P/'relkit/negative_l097.py').read_bytes()).hexdigest()
assert hashlib.sha256((P/'data/l095/ml-100k.zip').read_bytes()).hexdigest()==r['archive_sha256']
checked=0
with zipfile.ZipFile(P/'data/l095/ml-100k.zip') as z:
 for fold in range(1,6):
  base=np.loadtxt(io.BytesIO(z.read(f'ml-100k/u{fold}.base')),dtype=int);test=np.loadtxt(io.BytesIO(z.read(f'ml-100k/u{fold}.test')),dtype=int)
  observed={};truth={}
  for u,i,_,_ in base:observed.setdefault(int(u)-1,set()).add(int(i)-1)
  for u,i,rating,_ in test:
   if rating>=4:truth.setdefault(int(u)-1,set()).add(int(i)-1)
  for seed in range(3):
   runs=[x for x in r['runs'] if x['fold']==fold and x['seed']==seed]
   assert len({x['initial_sha256'] for x in runs})==1
   for run in runs:
    assert run['negative_draws']==run['positive_rows']*10 and run['known_positive_collisions']==0
    assert run['positive_rows']==sum(base[:,2]>=4) and len(run['loss_by_epoch'])==10
    path=P/'results/l097'/run['records_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==run['records_sha256']
    rows=json.loads(path.read_text());assert {x['user'] for x in rows}==set(truth)
    for row in rows:
     u=row['user'];rel=truth[u];assert set(row['relevant'])==rel
     candidates=set(range(1682))-observed.get(u,set());assert row['full_candidates']==len(candidates)
     neg=np.array(sorted(candidates-rel));rng=np.random.default_rng((970+fold)*10000+u)
     sampled=rel|set(rng.choice(neg,min(99,len(neg)),replace=False));assert len(sampled)==row['sampled_candidates']
     ideal=sum(1/math.log2(rank+1) for rank in range(1,min(10,len(rel))+1))
     for label,pool in [('full',candidates),('sampled',sampled)]:
      top=row[label+'_top'];assert len(top)==len(set(top))==min(10,len(pool)) and set(top)<=pool
      recall=len(set(top)&rel)/len(rel);ndcg=sum((i in rel)/math.log2(rank+1) for rank,i in enumerate(top,1))/ideal
      assert math.isclose(recall,row[label+'_recall'],abs_tol=1e-14) and math.isclose(ndcg,row[label+'_ndcg'],abs_tol=1e-14)
     assert row['sampled_ndcg']+1e-14>=row['full_ndcg']
     checked+=1
    for key in ['full_recall','full_ndcg','sampled_recall','sampled_ndcg']:
     assert math.isclose(sum(x[key] for x in rows)/len(rows),run[key],abs_tol=1e-14)
    assert run['evaluated_users']==len(rows) and run['excluded_users']==943-len(rows)
for arm in ['uniform','degree','hard']:
 for key in ['full_recall','full_ndcg','sampled_recall','sampled_ndcg']:
  folds=[sum(x[key] for x in r['runs'] if x['arm']==arm and x['fold']==f)/3 for f in range(1,6)]
  assert np.allclose(folds,r['summary'][arm][key]['fold_means'],rtol=0,atol=1e-14)
  assert np.isclose(np.std(folds,ddof=1),r['summary'][arm][key]['fold_sd'])
report={'status':'PASS','full_fits':45,'user_metric_records':checked,'initializations':'paired across arms','candidates':'independently reconstructed','metrics':'independently recomputed from saved rankings','ranking_optimality':'NOT_CHECKED_FROM_SCORES; score matrices are regenerated, not archived','source_hash':'MATCH'}
(P/'_audit_l097_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
