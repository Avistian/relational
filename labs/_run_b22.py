"""Execute the full approved course experiment; do not select favorable episodes."""
import json,platform
from pathlib import Path
import numpy as np
from relkit.refinement_b22 import experiment
P=Path(__file__).resolve().parent;E=P/'evidence/b22'
if __name__=='__main__':
 r=experiment();(E/'diagnostic.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
 (E/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,dtype='float64',rng='NumPy default_rng PCG64'),indent=2)+'\n')
 print(r['status'],len(r['conditions']),'trajectories',len(r['paired']),'paired effects')
 for mode in ['identity','skip','permute']:
  for seed in r['seeds']:
   d=[x['delta_ce'] for x in r['paired'] if x['seed']==seed and x['mode']==mode]
   print(mode,seed,'mean CE delta',float(np.mean(d)),'positive',sum(x>0 for x in d),'/',len(d))
