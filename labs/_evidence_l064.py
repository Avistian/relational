"""Reconstruct L064 data identities and metrics without importing its model.

Usage: python labs/_evidence_l064.py [evidence.json [report.json]]
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
ROOT=Path(__file__).resolve().parents[1];path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'labs/_verify_l064_v2_results.json'
r=json.loads(path.read_text());cases=[];datasets={};count=0
for row in r['records']:
 name=row['dataset']
 if name not in datasets:
  if name=='wdbc':b=load_breast_cancer();x=b.data.astype('float32');y=b.target
  else:
   d=pd.read_parquet(ROOT/f'labs/data/cache/{name}.parquet');key='class' if name=='diabetes' else 'Class';raw=d.pop(key);x=d.to_numpy(dtype='float32')
   y=(raw.astype(str)==sorted(raw.astype(str).unique())[-1]).astype(int).to_numpy() if raw.dtype==object or str(raw.dtype)=='category' else raw.astype(int).to_numpy()
  _,y=np.unique(y,return_inverse=True);datasets[name]=(x,y)
 x,y=datasets[name];seed=row['seed'];ids=np.arange(len(y));cap=r['config']['cap']
 if cap and cap<len(ids):ids,_=train_test_split(ids,train_size=cap,stratify=y,random_state=seed+6400)
 tr,te=train_test_split(ids,test_size=.5,stratify=y[ids],random_state=seed)
 assert np.array_equal(tr,row['train_ids']) and np.array_equal(te,row['test_ids'])
 assert not set(tr)&set(te)
 labels=y[tr]
 if row['condition']=='shuffled':labels=labels[np.random.default_rng(seed+64000).permutation(len(tr))]
 assert np.array_equal(labels,row['context_labels']) and np.array_equal(y[te],row['targets'])
 logits=np.array(row['trace']['logits']);p=np.array(row['probabilities']);ref=softmax(logits/.9,axis=1);delta=float(abs(ref-p).max());assert delta<4e-7
 loss=float(-np.log(p[np.arange(len(te)),y[te]]).mean());n1=sum(y[te]);n0=len(te)-n1
 auc=float((rankdata(p[:,1])[y[te]==1].sum()-n1*(n1+1)/2)/(n1*n0))
 assert abs(loss-row['log_loss'])<2e-7 and abs(auc-row['auc'])<1e-14
 keep=((x[tr[:1]]==x[tr]).mean(0)<1);assert np.array_equal(np.where(keep)[0],row['trace']['kept_columns'])
 cases.append(dict(dataset=name,seed=seed,condition=row['condition'],probability_delta=delta,loss_delta=abs(loss-row['log_loss']),auc_delta=abs(auc-row['auc'])));count+=len(te)
expected=len(r['config']['datasets'])*len(r['config']['seeds'])*2
assert len(cases)==expected
report=dict(status='PASS',records=len(cases),predictions=count,evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Independent raw dataset, split, label intervention, constant selection, softmax, NLL and rank-AUC reconstruction; no lesson model import',cases=cases)
output=ROOT/'reviews/lesson-quality-audit-047-070/064-evidence.json' if len(sys.argv)<3 else Path(sys.argv[2]);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
