"""Independent SQLite graph oracle and integer-count slice reconstruction."""
import json,sqlite3,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;start=time.perf_counter();s=json.loads((P/'evidence/l114/summary.json').read_text());z=np.load(P/'evidence/l114/analysis-inputs.npz');y=z['labels'];e=z['edge'];n=len(y)
db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE node(id INTEGER PRIMARY KEY,label INTEGER,known INTEGER); CREATE TABLE edge(a INTEGER,b INTEGER,PRIMARY KEY(a,b)) WITHOUT ROWID;')
train=set(z['train'].tolist());db.executemany('INSERT INTO node VALUES(?,?,?)',((i,int(v),int(i in train)) for i,v in enumerate(y)))
for first in range(0,e.shape[1],100000):
 pairs=e[:,first:first+100000].T.tolist();db.executemany('INSERT OR IGNORE INTO edge VALUES(?,?)',((a,b) for a,b in pairs if a!=b));db.executemany('INSERT OR IGNORE INTO edge VALUES(?,?)',((b,a) for a,b in pairs if a!=b))
degree=np.zeros(n,dtype=np.int64);h=np.full(n,np.nan);coverage=np.full(n,np.nan)
for i,d,hom,known in db.execute('SELECT a,COUNT(*),AVG(u.label=v.label),AVG(v.known) FROM edge JOIN node u ON u.id=a JOIN node v ON v.id=b GROUP BY a'):
 degree[i]=d;h[i]=hom;coverage[i]=known
np.testing.assert_array_equal(degree,z['degree']);np.testing.assert_allclose(h,z['homophily'],rtol=0,atol=0,equal_nan=True);np.testing.assert_allclose(coverage,z['train_neighbor_fraction'],rtol=0,atol=0,equal_nan=True)
# Independent binning (no calls to canonical masks or metrics).
deg_names=np.array(['0','1–2','3–5','6–10','11–20','21–50','51+'])
deg_group=deg_names[np.digitize(degree,[1,3,6,11,21,51])]
hom_names=np.array(['0–<.25','.25–<.50','.50–<.75','.75–1'])
hom_group=np.full(n,'undefined',dtype='<U12');ok=np.isfinite(h);hom_group[ok]=hom_names[np.digitize(h[ok],[.25,.5,.75])]
family={'degree':deg_group,'homophily':hom_group,'class':y.astype(str),'year':z['year'].astype(str)}
checked=0
for row in s['slice_rows']:
 ids=z[row['population']];ids=ids[family[row['family']][ids]==row['slice']];assert len(ids)==row['n']
 if not len(ids):assert row['gcn'] is None;continue
 for seed in range(10):
  a=z['gcn'][seed,ids]==y[ids];b=z['mlp'][seed,ids]==y[ids];pairs=np.bincount(a.astype(int)*2+b.astype(int),minlength=4)
  assert row['discordance'][seed]==[int(pairs[3]),int(pairs[2]),int(pairs[1]),int(pairs[0])]
  assert row['gcn'][seed]==int(pairs[2]+pairs[3])/len(ids)
  assert row['mlp'][seed]==int(pairs[1]+pairs[3])/len(ids)
  np.testing.assert_allclose(row['delta_pp'][seed],100*(int(pairs[2])-int(pairs[1]))/len(ids),atol=1e-12,rtol=0)
  checked+=1
eligible=[r for r in s['slice_rows'] if r['population']=='valid' and r['family']!='year' and r['n']>=200]
ranked=sorted(eligible,key=lambda r:(r['mean_delta_pp'],r['family'],r['slice']));assert ranked[0]==s['selected_validation_slice']
assert s['same_rule_on_test']['slice']==ranked[0]['slice'] and s['same_rule_on_test']['family']==ranked[0]['family']
r={'status':'PASS','sqlite_nodes':n,'unique_directed_nonself_edges':int(degree.sum()),'graph_arrays':'EXACT independent SQL joins/grouping','slice_seed_checks':checked,'selection':'validation only, support>=200, exact frozen test rule','seconds':time.perf_counter()-start}
(P/'_audit_l114_results.json').write_text(json.dumps(r,indent=2));print(r)
