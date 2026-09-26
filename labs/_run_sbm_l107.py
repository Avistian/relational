import argparse,sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
import torch
from sbm_l107 import load_sbm,train_sbm
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['H','O'],default='H');p.add_argument('--device',default='cpu');p.add_argument('--pilot',action='store_true');p.add_argument('--output',default=str(P/'evidence/l107/sbm'));p.add_argument('--data',default=str(P/'data/l107'));p.add_argument('--max-seconds',type=int,default=10000);a=p.parse_args()
 torch.set_num_threads(1);d=load_sbm(a.data);print({k:d[k] for k in ['rows','n','f','last']},flush=True)
 out=Path(a.output)/a.variant
 identity={n:hashlib.sha256((P/'relkit'/n).read_bytes()).hexdigest() for n in ['snapshot_l107.py','sbm_l107.py']};identity.update({'pilot':a.pilot,'device':a.device,'max_seconds':a.max_seconds})
 out.mkdir(parents=True,exist_ok=True)
 if (out/'identity.json').exists():raise RuntimeError('Output already exists; use a fresh directory to preserve previous evidence')
 (out/'identity.json').write_text(json.dumps(identity,indent=2))
 if (out/'result.json').exists():raise RuntimeError('Completed or incomplete run already exists; use a fresh directory, no unauthenticated resume')
 train_sbm(d,a.variant,out,a.device,epochs=1 if a.pilot else 100,max_seconds=a.max_seconds,pilot=a.pilot)
