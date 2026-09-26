"""Fresh full-data MLP training: smoke2 epochs; closer50; paper10x500."""
import argparse,json
from pathlib import Path
import torch
import numpy as np
from relkit.ogb_l112 import load_arxiv
from relkit.error_l114 import train_mlp,TARGETS
p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='smoke');p.add_argument('--device',default='cpu');p.add_argument('--data',default='labs/data/l112');p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(1);x,edge,y,split,audit=load_arxiv(a.data)
records=[]
for seed in (range(10) if a.preset=='paper' else [100]):
 r=train_mlp(x,y,split,seed,{'smoke':2,'closer':50,'paper':500}[a.preset],Path(a.output)/f'seed-{seed}',a.device);records.append(r);print({k:v for k,v in r.items() if k!='history'})

if a.preset=='paper':
 for pop,target in TARGETS.items():
  values=np.array([r['scores'][pop]*100 for r in records]);print({'population':pop,'mean_percent':float(values.mean()),'sample_sd_pp':float(values.std(ddof=1)),'target':target,'verdict':'CLOSE' if abs(values.mean()-target)<=.5 else 'OUTSIDE_TOLERANCE'})
