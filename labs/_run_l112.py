"""Executable full-data teaching or ten-seed paper schedule."""
import argparse,json
from pathlib import Path
import torch
from relkit.ogb_l112 import load_arxiv,normalized_adjacency,train_run
p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','paper'],default='smoke');p.add_argument('--device',default='cpu');p.add_argument('--data',default='labs/data/l112');p.add_argument('--output',required=True);p.add_argument('--seeds',default=None);a=p.parse_args()
torch.set_num_threads(1);x,edge,y,split,audit=load_arxiv(a.data);adj=normalized_adjacency(edge,len(y));out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'data-audit.json').write_text(json.dumps(audit,indent=2))
seeds=list(map(int,a.seeds.split(','))) if a.seeds else (list(range(10)) if a.preset=='paper' else [100])
for seed in seeds:
 r=train_run(x,adj,y,split,seed,500 if a.preset=='paper' else 2,out/f'seed-{seed}',a.device);print({k:v for k,v in r.items() if k!='history'})
