"""Run all 100 seeds of the full fixed-split Cora reconstruction by default."""
import argparse,json,hashlib,platform
from pathlib import Path
import numpy as np
import torch,torch_geometric
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from gcn_l082 import load_cora
from pyg_l086 import as_data,train_citation
p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--output',default='_paper_l086_results.json');a=p.parse_args()
if a.seeds<1:p.error('positive seed count required')
root=Path(__file__).resolve().parent;torch.set_num_threads(1);data=as_data(load_cora(root));runs=[]
for seed in range(a.seeds):
 r=train_citation(data,seed);runs.append(r);print(seed,r['test_accuracy'],r['epochs'],flush=True)
acc=np.array([r['test_accuracy'] for r in runs])
r={'status':'EXECUTED_FULL_100_SEED_RECONSTRUCTION' if a.seeds==100 else 'EXECUTED_PARTIAL','experiment':'Fey-Lenssen Table 1 GCN Cora fixed; PyG 1.2.0 protocol port','paper_mean':.815,'paper_sd':.006,'mean':float(acc.mean()),'sample_sd':float(acc.std(ddof=1)) if len(acc)>1 else None,'runs':runs,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__},'historical_parity':'INCOMPARABLE: modern kernels and locally declared seeds; 1.2.0 is an auditable release proxy, not a confirmed paper-run commit','sha256':{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in ['relkit/pyg_l086.py','relkit/gcn_l082.py','_sources_l078.json','_sources_l086.json']}}
out=Path(a.output);out=out if out.is_absolute() else root/out;out.write_text(json.dumps(r,indent=2)+'\n');print(r['mean'],r['sample_sd'])
