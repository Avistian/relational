"""Run named release replay; smoke/closer are explicitly different experiments."""
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch
from relkit.tgat_l103 import load_wikipedia,run_training
P=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='smoke');p.add_argument('--seeds',default='0');p.add_argument('--device',default='cpu');p.add_argument('--data',default=str(P/'data/l102'));p.add_argument('--output',default=str(P/'results/l103'));a=p.parse_args()
 torch.set_num_threads(1);n,e,d,audit=load_wikipedia(a.data)
 budgets={'smoke':(400,120,1),'closer':(6000,600,3),'paper':(None,None,50)};tr,ev,epochs=budgets[a.preset]
 if tr:d={key:({f:x[:(tr if key=='train' else ev)] for f,x in value.items()} if key!='full' else value) for key,value in d.items()}
 sha=hashlib.sha256((P/'relkit/tgat_l103.py').read_bytes()).hexdigest()
 identity={'implementation_sha256':sha,'data':audit,'preset':a.preset,'epochs_max':epochs,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'device':a.device,'platform':platform.platform()}
 root=Path(a.output)/a.preset;root.mkdir(parents=True,exist_ok=True)
 path=root/'identity.json'
 if path.exists():assert json.loads(path.read_text())==identity,'Run identity changed; use a fresh output directory'
 else:path.write_text(json.dumps(identity,indent=2))
 records=[]
 for seed in map(int,a.seeds.split(',')):
  out=root/f'seed-{seed}'
  records.append(json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else run_training(n,e,d,seed=seed,epochs=epochs,output=out,device=a.device))
 report={'status':'COMPLETE','preset':a.preset,'identity':identity,'records':records,'paper_comparison':'NOT_ESTABLISHED' if a.preset=='paper' else 'INCOMPARABLE'}
 (P/f'_verify_l103_{a.preset}_results.json').write_text(json.dumps(report,indent=2)+'\n')
