"""Fresh complete pretrained model panel. No historical result rewriting."""
import argparse,hashlib,json,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import torch
from relkit import driftpfn_l068_v2 as c
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preset',choices=c.PRESETS68,default='closer');parser.add_argument('--output',default='_verify_l068_v2_results.json');args=parser.parse_args()
 torch.set_num_threads(1)
 r=c.run_experiment(ROOT,c.PRESETS68[args.preset]);r['operator_sha256']=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest();r['executor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (ROOT/args.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(r['seconds'])
