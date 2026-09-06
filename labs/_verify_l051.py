"""Regenerate measured L051 evidence; CPU threads deliberately limited."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from relkit.bias_interventions import run_experiment
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',default='lab');p.add_argument('--dataset',action='append');args=p.parse_args()
    with threadpool_limits(limits=1):result=run_experiment(args.preset,args.dataset)
    root=Path(__file__).parent
    result['source_sha256']={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['relkit/bias_interventions.py','relkit/checkpoint.py']}
    out=root/('_verify_l051_results.json' if args.preset=='lab' else f'_paper_repro_l051_{args.preset}_summary.json')
    out.write_text(json.dumps(result,indent=2));print(out,result['elapsed_s'])
