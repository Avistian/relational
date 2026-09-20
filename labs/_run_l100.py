"""Fresh complete checkpoint run; all data and 24 fits, no shortened preset."""
import hashlib,json
from pathlib import Path
from relkit.checkpoint_l100 import run_suite
P=Path(__file__).resolve().parent
if __name__=='__main__':
 r=run_suite(P/'data/l099',P/'_experiment_l100_results.json')
 r['implementation_sha256']=hashlib.sha256((P/'relkit/checkpoint_l100.py').read_bytes()).hexdigest()
 (P/'_experiment_l100_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['summary'])
