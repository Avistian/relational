"""Bounded fresh training; artifact is separate from student output."""
import hashlib,json
from pathlib import Path
from relkit.vime_l071 import run_experiment
if __name__=='__main__':
    root=Path(__file__).parent
    result=run_experiment()
    result['implementation_sha256']=hashlib.sha256((root/'relkit/vime_l071.py').read_bytes()).hexdigest()
    (root/'_verify_l071_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
