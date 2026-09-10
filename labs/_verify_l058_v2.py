"""Run the complete historical-panel and held-out-method audit, CPU only."""
import argparse, hashlib, json, importlib.metadata, time
from pathlib import Path
from relkit.talent_audit_l058 import audit
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=1000);p.add_argument('--output',type=Path,default=ROOT/'_verify_l058_v2_results.json');args=p.parse_args()
 start=time.perf_counter();result=audit(ROOT,args.trials)
 result['operator_sha256']=hashlib.sha256((ROOT/'relkit/talent_audit_l058.py').read_bytes()).hexdigest()
 result['versions']={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']}
 result['seconds']=time.perf_counter()-start
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print({k:result[k] for k in ['mean_ranks','paired_gaps','tiny_trials','seconds']})
