import hashlib,json,os
from pathlib import Path
from relkit.realmlp import RealMLPS
from relkit.realmlp_experiment import run_suite
from _fetch_l052 import fetch

if __name__=='__main__':
    root=Path(__file__).resolve().parent;os.chdir(root);fetch()
    result=run_suite(RealMLPS)
    result['code_hashes']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['relkit/realmlp.py','relkit/realmlp_experiment.py']}
    (root/'_verify_l053_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['ranks'],indent=2));print('Seconds',result['elapsed_seconds'])
