"""Execute all 24 fits; no pilot or silent shortened preset."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
from compare_l099 import run_suite
if __name__=='__main__':
 r=run_suite(P/'data/l099',P/'_experiment_l099_results.json')
 r['implementation_sha256']=hashlib.sha256((P/'relkit/compare_l099.py').read_bytes()).hexdigest()
 (P/'_experiment_l099_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['summary'])
