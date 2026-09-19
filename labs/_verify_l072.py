import json
from pathlib import Path
from relkit.contrastive_l072 import run_experiment
if __name__=='__main__':
    result=run_experiment()
    Path(__file__).with_name('_verify_l072_results.json').write_text(json.dumps(result,indent=2))
    for row in result['summary']:print(row['dataset'],row['arm'],round(row['mean'],4),round(row['sd'],4))
    print('seconds',result['elapsed_seconds'])
