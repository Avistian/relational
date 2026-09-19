import hashlib,json
from pathlib import Path
from relkit.ssl_regimes_l073 import run_regimes
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    result=run_regimes()
    result['source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['relkit/ssl_regimes_l073.py','relkit/contrastive_l072.py']}
    (ROOT/'_verify_l073_results.json').write_text(json.dumps(result,indent=2))
    print(result['crossings']);print('Seconds:',result['elapsed_seconds'])
