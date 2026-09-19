"""Run paired target-table transfers with explicit checkpoint and data identity."""
import hashlib,json,os
from pathlib import Path
from relkit.carte_l074 import run_transfer
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(ROOT);r=run_transfer()
    r['source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['relkit/carte_l074.py','data/l074/manifest.json']}
    (ROOT/'_verify_l074_results.json').write_text(json.dumps(r,indent=2));print(r['summary'])
