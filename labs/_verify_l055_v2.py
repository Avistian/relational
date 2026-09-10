"""Rebuild local or larger L055 evidence; no test-score selection."""
import argparse, hashlib, json
from pathlib import Path
from _fetch_l055 import fetch
from _check_release_l055 import verify_extracted
from relkit.temporal_experiment_v2 import run_suite, summarize

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preset',choices=['smoke','windows-smoke','local','closer'],default='local')
    args=parser.parse_args()
    config = {'smoke':dict(names=('sberbank-housing',),seeds=(0,),train_cap=300,eval_cap=150,epochs=2,trees=10),
              'windows-smoke':dict(names=('sberbank-housing',),seeds=(0,),split_ids=(0,1,2),train_cap=300,eval_cap=150,epochs=2,trees=10),
              'local':{}, 'closer':dict(train_cap=6000,eval_cap=2000,epochs=64,trees=300,split_ids=(0,1,2))}[args.preset]
    root=fetch();verify_extracted(root)
    result=run_suite(root=root,**config);result['summary']=summarize(result)
    result['source_hashes']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['relkit/temporal.py','relkit/temporal_v2.py','relkit/temporal_experiment_v2.py','relkit/tabm_v2.py','_fetch_l055.py']}
    path=ROOT/('_verify_l055_v2_results.json' if args.preset=='local' else f'_l055_v2_{args.preset}_results.json')
    path.write_text(json.dumps(result,indent=1));print(path, result['seconds'])
