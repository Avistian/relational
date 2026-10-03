"""Run the frozen course diagnostic; preserve every keyed prediction."""
import json
from pathlib import Path
from relkit.synthetic_b13 import run_experiment
if __name__=='__main__':
    out=Path(__file__).resolve().parent/'evidence/b13/diagnostic.json'
    result=run_experiment();encoded=json.dumps(result,indent=2)+'\n'
    if out.exists() and out.read_text()!=encoded:raise SystemExit('Immutable result differs; investigate before replacement')
    out.write_text(encoded)
    print([{k:r[k] for k in ['seed','diversity','arm','mse','corrupted_mse']} for r in result['conditions']])
