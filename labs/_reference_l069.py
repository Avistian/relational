"""Precompute independent original-source probabilities for the declared L069 panel.

No lesson implementation or saved lesson predictions are read. Source CSVs,
predeclared row partitions and historical TabPFN 2.0.9 determine every case.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from _evidence_l069 import original_model,source_predict
ROOT=Path(__file__).resolve().parent
DATA={'cmc':'f1954d85d813fb0586cbc27cfe407e40e238ea0b39a3cf5c652694c1f4225d95','winequality-red':'65fc57ebcf15bf3264b2e00212539a2c0840f1d6abf6ae9e8f018d58030b4b29','winequality-white':'b8f4937cecb06e57ba75f5b851e1d0376b3d507d1743d5409cd20bd920120f93','iris':'b5bb2427eb7ff7ca54f3d1ad6fe52610caf38b1af3a6bfaf965dec5d63af2d6d'}
def run(output):
 output=Path(output);assert not output.exists();journal=output.with_suffix('.partial.jsonl');assert not journal.exists();start=time.perf_counter();model,adapter=original_model();records=[];datasets={}
 for name,digest in DATA.items():
  file=ROOT/'data/cache/l069-source'/name/(name+'.csv');assert hashlib.sha256(file.read_bytes()).hexdigest()==digest;df=pd.read_csv(file);target='label' if 'label' in df else df.columns[-1];datasets[name]=(df.drop(columns=target).to_numpy(dtype='float32'),pd.Categorical(df[target]).codes.astype(int))
 def predict(sid,condition,cx,cy,qx,train,test):
  p,classes=source_predict(model,cx,cy,qx);records.append(dict(split_id=sid,condition=condition,context_ids=np.asarray(train).tolist(),query_ids=np.asarray(test).tolist(),classes=classes.tolist(),probabilities=p.tolist()));
  with journal.open('a') as f:f.write(json.dumps(records[-1],allow_nan=False)+'\n')
  print('SOURCE',sid,condition,len(test),flush=True)
 for axis,names in [('novelty',['cmc','winequality-red','winequality-white']),('features',['iris','cmc','winequality-red'])]:
  for name in names:
   x,y=datasets[name];ids=np.arange(len(y))
   for seed in [42,2023,789]:
    if axis=='novelty':
     for held in np.unique(y):
      novel=ids[y==held];known=ids[y!=held];strata=y[known] if len(novel)>=len(np.unique(y[known])) else None;known_test,train=train_test_split(known,train_size=len(novel),random_state=seed,stratify=strata);test=np.r_[novel,known_test];predict(f'novelty:{name}:{seed}:{held}','leave-one-class-out',x[train],y[train],x[test],train,test)
     train,test=train_test_split(ids,test_size=.2,random_state=seed,stratify=y);train=train[y[train]!=0];predict(f'natural:{name}:{seed}','natural-prevalence-class0',x[train],y[train],x[test],train,test)
    else:
     train,test=train_test_split(ids,test_size=.2,random_state=seed);order=np.random.default_rng(seed).permutation(x.shape[1])
     for level in [0,.2,.4,.6,.8,1.]:
      columns=order[:math.floor(level*x.shape[1])];query=x[test].astype(float);query[:,columns]=x[train].astype(float)[:,columns].mean(0);predict(f'features:{name}:{seed}',f'{level:.0%}',x[train],y[train],query,train,test)
 for seed in [42,2023,789]:
  rng=np.random.default_rng(seed);cx=rng.normal(size=(256,2));qx=rng.normal(size=(512,2));cy=(cx[:,0]>0).astype(int)
  for kind in ['iid','covariate','concept']:
   q=qx.copy()
   if kind=='covariate':q[:,0]+=1.5
   predict(f'shift:{seed}:{kind}',kind,cx,cy,q,np.arange(256),np.arange(256,768))
 result=dict(status='PASS',source='tabpfn2.0.9',checkpoint_sha256=hashlib.sha256((ROOT/'data/cache/foundation/tabpfn-v2.ckpt').read_bytes()).hexdigest(),data_sha256=DATA,attention_adapter_checks=adapter,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((ROOT/'_evidence_l069.py').read_bytes()).hexdigest(),records=records,seconds=time.perf_counter()-start)
 with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print('WROTE',output,'seconds',result['seconds'],flush=True)
if __name__=='__main__':
 import sys
 run(sys.argv[1])
