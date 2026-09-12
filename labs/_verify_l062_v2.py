"""New L062 measurements; never overwrite historical evidence or resume silently."""
import argparse,json
from pathlib import Path
import torch
from relkit.tabpfn_l062_v2 import run_experiment
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','lab','closer'],default='lab');p.add_argument('--output',type=Path,default=ROOT/'_verify_l062_v2_results.json');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Choose a fresh output; no resume/overwrite')
 torch.set_num_threads(1);r=run_experiment(ROOT,a.preset);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
