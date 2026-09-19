"""Full published Cora port plus separate inductive checkpoint extension."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch,scipy
from relkit.gcn_l082 import run_cora,load_cora
from relkit.checkpoint_l090 import train_inductive,verdict
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['paper','inductive'],default='paper');p.add_argument('--preset',choices=['smoke','closer','paper'],default='paper');p.add_argument('--output');a=p.parse_args()
root=Path(__file__).resolve().parent;started=time.time()
if a.lane=='paper':
    n={'smoke':1,'closer':10,'paper':100}[a.preset];r=run_cora(n,root)
    r['status']='COMPLETE' if n==100 else 'INCOMPLETE';r['course_verdict']=verdict(r['mean'],n,True)
    r['historical_exact_parity']='INCOMPARABLE';r['tolerance']=.01
else:
    data=load_cora(root);runs=[]
    for seed in ([0] if a.preset=='smoke' else [0,1,2]):
        result=train_inductive(data,seed,epochs=3 if a.preset=='smoke' else 100);runs.append(result)
        print('inductive',seed,result['test_accuracy'],flush=True)
    scores=np.array([r['test_accuracy'] for r in runs]);r={'status':'COMPLETE','experiment':'Cora inductive teaching extension','runs':runs,'mean':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)) if len(scores)>1 else None,'paper_comparison':'INCOMPARABLE'}
r['preset']=a.preset;r['seconds']=time.time()-started;r['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'machine':platform.machine(),'threads':torch.get_num_threads()}
r['sha256']={str(q.relative_to(root)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [root/'relkit/gcn_l082.py',root/'relkit/checkpoint_l090.py',root/'_sources_l078.json',Path(__file__)]}
out=root/(a.output or f'_{a.lane}_l090_results.json');out.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='runs'})
