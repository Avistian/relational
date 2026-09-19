"""Run the named QM9 GG-NN reconstruction; see l081-reproduction.md."""
import argparse,json
from relkit.qm9_l081 import run_reconstruction
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper-budget'],default='smoke');p.add_argument('--seed',type=int,default=81);p.add_argument('--data',default='labs/data/cache/l081');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();r=run_reconstruction(a.data,a.output,a.preset,a.seed,a.device);print(json.dumps({k:r[k] for k in ['status','test_mae_debye','seconds']},indent=2))
