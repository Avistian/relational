"""Run the explicit L061 v2 PFN experiment with a fresh evidence path."""
import argparse,time
from relkit.pfn_l061_v2 import run_experiment
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preset',default='lab',choices=['smoke','lab','closer','paper']);p.add_argument('--output');p.add_argument('--device',default='cpu');a=p.parse_args()
 run_experiment(a.preset,a.output or f'labs/data/cache/l061-rerun-{time.time_ns()}.json',a.device)
