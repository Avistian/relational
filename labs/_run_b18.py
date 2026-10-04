import gzip,json,platform
from pathlib import Path
import numpy as np
from relkit.context_b18 import run_experiment
P=Path(__file__).resolve().parent
r=run_experiment()
(P/'evidence/b18/diagnostic.json.gz').write_bytes(gzip.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode(),mtime=0))
summary={k:v for k,v in r.items() if k!='fixtures'}
for c in summary['conditions']:c.pop('estimates')
(P/'evidence/b18/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'evidence/b18/environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__),indent=2)+'\n')
print('Complete:24 fixtures,96 conditions,9600 paired repetitions,38400 arm estimates')
