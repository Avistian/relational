"""Bounded CPU evidence; writes only the explicit output path."""
import argparse,hashlib,json,time,platform,importlib.metadata
from pathlib import Path
from threadpoolctl import threadpool_limits
from relkit.validation_audit_l059 import experiment,null_reanalysis
ROOT=Path(__file__).resolve().parent

def run(output,repetitions=30):
    start=time.time()
    with threadpool_limits(1):result=experiment(repetitions=repetitions)
    old=ROOT/'_verify_l059_results.json'
    result['historical_null']=null_reanalysis(json.loads(old.read_text())['rows'])
    result['provenance']={'historical_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
      'operator_sha256':hashlib.sha256((ROOT/'relkit/validation_audit_l059.py').read_bytes()).hexdigest(),
      'python':platform.python_version(),'libraries':{p:importlib.metadata.version(p) for p in ['numpy','scipy']},
      'wall_seconds':time.time()-start,'source_manifest':'_sources_l059_v2.json'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'means':result['means'],'summary':result['summary'],'seconds':result['provenance']['wall_seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repetitions',type=int,default=30);a=p.parse_args();run(a.output,a.repetitions)
