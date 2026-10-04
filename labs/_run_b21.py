"""Fresh finite experiment: full fixed grid, no paid compute."""
import json,time,platform
from pathlib import Path
import numpy as np
import torch
from relkit.structural_b21 import experiment
P=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(1);start=time.monotonic();report=experiment()
    (P/'evidence/b21/diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    (P/'evidence/b21/runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,paid_usd=0),indent=2)+'\n')
    print('COMPLETE:',len(report['states']),'seed/states;',len(report['conditions']),'conditions')
    for r in report['conditions']:
        if r['budget']==2:print(r['seed'],r['method'],'cost',r['cost'],'loss',round(r['loss'],6),'regret',round(r['regret'],6))
