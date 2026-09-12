"""Fresh scoped L063 evidence; never overwrite historical or existing outputs."""
import argparse,hashlib,json
from pathlib import Path
from relkit.scm_l063_v2 import run_experiment
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','lab','closer'],default='lab');p.add_argument('--output',type=Path,default=Path(__file__).with_name('_verify_l063_v2_results.json'));a=p.parse_args()
 result=run_experiment(a.preset,a.output)
 result['reference_source_sha256']=hashlib.sha256(Path(__file__).with_name('relkit').joinpath('scm_l063_v2.py').read_bytes()).hexdigest()
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':result['summary'],'seconds':result['elapsed_seconds'],'oracle_nll':result['finite_oracle']['correct_nll'],'ignored_query_x_nll':result['finite_oracle']['ignored_query_x_nll']},indent=2))
