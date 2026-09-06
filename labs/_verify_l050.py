"""Run the actual three-task checkpoint and save full selection/prediction evidence."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from relkit.checkpoint import reference_parity,run_comparison
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        parity=reference_parity()
        result=run_comparison()
    result['reference_parity']=parity
    Path(__file__).with_name('_verify_l050_results.json').write_text(json.dumps(result,indent=2))
    print(result['statistics'])
