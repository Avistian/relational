"""Independent grouping, scoring and path audit of immutable saved evidence.

Does not import the implementation. Uses pairwise column equality rather than
its typed-key helper; recomputes selected probabilities and all test metrics.
"""
import itertools,json,math
from pathlib import Path
P=Path(__file__).resolve().parent;r=json.loads((P/'evidence/b16/diagnostic.json').read_text())
assert len(r['subsets'])==72 and len(r['selections'])==27
worlds=['signal','xor','null'];cols=['A','B','K'];allcols=[list(s) for k in range(4) for s in itertools.combinations(cols,k)]
assert {(x['world'],x['penalty'],tuple(x['cols'])) for x in r['subsets']}=={(w,p,tuple(s)) for w in worlds for p in [0,.5,1] for s in allcols}
def oracle(tr,query,s):
 out=[]
 for q in query:
  matched=[t for t in tr if all(t[c]==q[c] for c in s)]
  used=matched or tr
  out.append(sum(t['y'] for t in used)/len(used))
 return out
for w in worlds:
 ds=r['datasets'][w]
 for split,prefix in [('train','tr'),('valid','va'),('test','te')]:
  rows=ds[split];assert len(rows)==8
  for i,row in enumerate(rows):
   assert row['id']==row['K']==f'{prefix}-{i}'
   assert row['A']==i//4 and row['B']==(i//2)%2
   assert row['y']==(i//4 if w=='signal' else int((i//4)!=((i//2)%2)) if w=='xor' else i%2)
for item in r['subsets']:
 ds=r['datasets'][item['world']];tr,va=ds['train'],ds['valid'];s=item['cols']
 groups=[];remaining=set(range(8))
 while remaining:
  first=min(remaining);group=[j for j in sorted(remaining) if all(tr[j][c]==tr[first][c] for c in s)]
  groups.append(group);remaining-=set(group)
 omega=sum(math.sqrt(len(g)) for g in groups)/8
 probs=oracle(tr,va,s);risk=sum((p-v['y'])**2 for p,v in zip(probs,va))/8
 assert item['probabilities']==probs and item['sizes']==[len(g) for g in groups]
 assert math.isclose(item['omega'],omega,abs_tol=1e-14) and math.isclose(item['risk'],risk,abs_tol=1e-14)
 assert math.isclose(item['J'],risk+item['penalty']*omega,abs_tol=1e-14)
for sel in r['selections']:
 ds=r['datasets'][sel['world']];grid={tuple(x['cols']):x for x in r['subsets'] if x['world']==sel['world'] and x['penalty']==sel['penalty']}
 if sel['method']=='exhaustive':
  minimum=min(x['J'] for x in grid.values());expected=next(s for s in allcols if grid[tuple(s)]['J']<=minimum+1e-12)
 else:
  current=cols[:] if sel['method']=='backward' else []
  while True:
   moves=([c for c in cols if c not in current] if sel['method']=='forward' else current)
   if not moves:break
   options=[sorted(current+[c]) if sel['method']=='forward' else [x for x in current if x!=c] for c in moves]
   best=min(options,key=lambda s:grid[tuple(s)]['J'])
   if grid[tuple(current)]['J']-grid[tuple(best)]['J']<=1e-12:break
   current=best
  expected=current
 assert sel['selected']==expected and abs(sel['J']-grid[tuple(expected)]['J'])<1e-14
 probs=oracle(ds['train'],ds['test'],expected)
 assert sel['predictions']==[dict(id=t['id'],probability=p,label=t['y']) for t,p in zip(ds['test'],probs)]
 assert sel['test_brier']==sum((p-t['y'])**2 for p,t in zip(probs,ds['test']))/8
assert len(r['graph_checks'])==24
for x in r['graph_checks']:
 tr=r['datasets'][x['world']]['train'];blockof={i:k for k,b in enumerate(x['blocks']) for i in b}
 assert sorted(blockof)==list(range(8))
 for i,j in itertools.product(range(8),repeat=2):assert (blockof[i]==blockof[j])==all(tr[i][c]==tr[j][c] for c in x['cols'])
assert len(r['test_interventions'])==256
assert {tuple(x['labels']) for x in r['test_interventions']}==set(itertools.product((0,1),repeat=8))
assert all(x['unchanged'] for x in r['test_interventions'])
assert r['feature_boundary']==dict(typed=[[0,1,2,3],[4,5,6,7]],anonymous=[list(range(8))],full=[[0,1],[2,3],[4,5],[6,7]])
out=dict(status='PASS',independent_subset_scores=72,independent_selections=27,keyed_test_predictions=216,graph_pairwise_relations=24*64,test_intervention_coverage=256)
(P/'_verify_b16_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
