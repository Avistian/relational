"""Fresh full named reproduction or clearly separate short teaching run."""
import argparse,hashlib,json,sys,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from relkit.checkpoint_l110 import load_wikipedia,run_training
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','paper'],default='smoke');p.add_argument('--arm',choices=['release','clean','both'],default='both');p.add_argument('--seeds',default='0');p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);nodes,edges,data,audit=load_wikipedia(P/'data/l102')
identity={'source_sha256':hashlib.sha256((P/'relkit/checkpoint_l110.py').read_bytes()).hexdigest(),'data':audit,'preset':a.preset,'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'device':a.device,'libraries':{k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn']}}
a.output.mkdir(parents=True,exist_ok=True);f=a.output/'identity.json'
if f.exists():assert json.loads(f.read_text())==identity,'Changed run identity: use a fresh directory'
else:f.write_text(json.dumps(identity,indent=2))
if a.preset=='smoke':data={k:({f:x[:(400 if k=='train' else 200)] for f,x in d.items()} if k!='full' else d) for k,d in data.items()}
for seed in map(int,a.seeds.split(',')):
 for arm in (['release','clean'] if a.arm=='both' else [a.arm]):
  out=a.output/arm;done=out/f'seed-{seed}.json'
  if done.exists():
   assert (out/f'seed-{seed}.pt').exists() and (out/f'seed-{seed}-predictions.npz').exists(),'Incomplete saved artifacts'
   print('Completed seed already present:',arm,seed);continue
  run_training(nodes,edges,data,seed=seed,epochs=50 if a.preset=='paper' else 2,output=out,device=a.device,arm=arm)
