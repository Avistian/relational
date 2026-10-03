"""Independent immutable-prediction rescore, complete matrix accounting and plot."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
raw_x,raw_y=load_diabetes(return_X_y=True,scaled=False)
P=Path(__file__).resolve().parent;E=P/'evidence/b09';rows=[];missing=[]
for model,seed in itertools.product(['tabfm','exaone','nori'],range(3)):
 path=E/f'{model}-{seed}.json'
 if not path.exists():missing.append(f'{model}/{seed}');continue
 r=json.loads(path.read_text());d=np.load(P/f'data/b09/split-{seed}.npz');a=np.load(E/f'{model}-{seed}-predictions.npz')
 expected_train,expected_test=train_test_split(np.arange(442),test_size=.2,random_state=seed)
 np.testing.assert_array_equal(d['train'],expected_train);np.testing.assert_array_equal(d['test'],expected_test)
 np.testing.assert_array_equal(d['x'],raw_x.astype('float32'));np.testing.assert_array_equal(d['y'],raw_y.astype('float32'))
 assert r['input_sha256']==hashlib.sha256((P/f'data/b09/split-{seed}.npz').read_bytes()).hexdigest()
 assert len(set(a['ids']))==89 and set(a['ids'])==set(d['test'])
 assert not set(d['test'])&set(d['train']) and len(d['train'])==353
 idx={int(k):float(v) for k,v in zip(a['ids'],a['predictions'])};pred=np.array([idx[int(k)] for k in d['test']]);truth=d['y'][d['test']].astype(float)
 score=float(np.sqrt(((pred-truth)**2).sum()/len(truth)))
 assert abs(score-r['rmse'])<1e-9 and len(r['warm_s'])==10
 assert all(x>0 for x in r['warm_s']) and np.isfinite(pred).all()
 r['baseline_rmse']=float(np.sqrt(np.mean((truth-d['y'][d['train']].mean())**2)))
 rows.append(r)
summary=[]
for model in ['tabfm','exaone','nori']:
 rs=[r for r in rows if r['model']==model]
 if len(rs)==3:
  summary.append(dict(model=model,rmse_mean=float(np.mean([r['rmse'] for r in rs])),rmse_sd=float(np.std([r['rmse'] for r in rs],ddof=1)),warm_median_s=float(np.median([r['warm_median_s'] for r in rs])),peak_rss_mib=max(r['peak_rss_mib'] for r in rs)))
missing_status={}
for key in missing:
 model=key.split('/')[0]
 if model=='tabfm' and (E/'resource-gate.json').exists():missing_status[key]='INCOMPLETE_RESOURCE_GATE'
 else:
  budget=json.loads((E/'budget.json').read_text())
  left=budget['cap_s']-budget['preparation_allowance_s']-budget['validation_reserve_s']-sum(x['seconds'] for x in budget['attempts'])
  missing_status[key]='INCOMPLETE_BUDGET_GATE' if left<1 else 'NOT_RUN_OR_FAILED: see budget attempts'
out=dict(missing_status=missing_status,status='COMPLETE' if len(rows)==9 else 'INCOMPLETE',completed=len(rows),expected=9,missing=missing,rows=rows,summary=summary,paper='INCOMPLETE_SOURCE_PROTOCOL',learner='PENDING_WRITTEN_DEFENSE')
(E/'course-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if rows:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.8))
 colors={'tabfm':'#167567','exaone':'#3874a5','nori':'#a15c21'}
 for ax,key,label in zip(axes,['rmse','warm_median_s','peak_rss_mib'],['RMSE ↓ (target units)','Warm median ↓ (seconds / 89 rows)','Peak RSS ↓ (MiB, absolute process)']):
  for i,model in enumerate(colors):
   rs=[r for r in rows if r['model']==model];ax.scatter([i+(r['seed']-1)*.13 for r in rs],[r[key] for r in rs],color=colors[model])
   for r in rs:ax.annotate(str(r['seed']),(i+(r['seed']-1)*.13,r[key]),xytext=(3,3),textcoords='offset points',fontsize=8)
  ax.set(xticks=range(3),xticklabels=list(colors),ylabel=label);ax.grid(axis='y',alpha=.2)
 fig.suptitle(f'Measured CPU course evidence · {len(rows)}/9 runs · labels are split seeds');fig.tight_layout();fig.savefig(P/'figures/b09/results.png',dpi=160);plt.close(fig)
