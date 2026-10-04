"""Run the frozen complete mechanism grid and preserve unrounded arrays."""
import json,platform
from pathlib import Path
import torch
from relkit.representations_b17 import run_experiment
P=Path(__file__).resolve().parent
if __name__=='__main__':
    result=run_experiment()
    (P/'evidence/b17/diagnostic.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (P/'evidence/b17/environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,device='cpu',threads=1),indent=2)+'\n')
    for seed in result['seeds']:
        print(seed['seed'],[(x['task'],x['mode'],round(x['prediction_delta'],6)) for x in seed['records']])
