"""Execute pinned upstream selector with a local log-loss adapter; not full AutoGluon."""
import ast,json,logging,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from relkit.cross_ensemble import greedy_select,binary_loss
ROOT=Path(__file__).resolve().parent
src=(ROOT/'sources/l057/ensemble_selection.py').read_text()
nodes=[n for n in ast.parse(src).body if isinstance(n,ast.ClassDef)]
ns=dict(np=np,pd=pd,time=time,Counter=Counter,logger=logging.getLogger('source'),PROBLEM_TYPES=['binary'])
exec(compile('from __future__ import annotations\n'+'\n\n'.join(ast.get_source_segment(src,n) for n in nodes),'<pinned upstream>','exec'),ns)
cls=ns['EnsembleSelection']
class LocalMetricAdapter(cls):
    def _calculate_regret(self,y_true,y_pred_proba,metric,sample_weight=None):return binary_loss(y_true,y_pred_proba)
rng=np.random.default_rng(57);cases=[]
for seed in range(5):
 y=rng.integers(0,2,120);p=rng.uniform(.05,.95,(120,3))
 local,trace=greedy_select(p,y,40)
 ref=LocalMetricAdapter(ensemble_size=40,problem_type='binary',metric=None).fit([p[:,j] for j in range(3)],y)
 gap=float(np.max(np.abs(local-ref.weights_)));assert gap<1e-12,(local,ref.weights_)
 cases.append(dict(case=seed,weight_max_abs_error=gap))
r=dict(status='MATCH',cases=cases,scope='Pinned selector classes executed; local binary log-loss metric adapter. Five deterministic non-tied fixtures. Upstream rounding/tie rules differ; no universal or training parity claim.',sha256=hashlib.sha256(src.encode()).hexdigest())
(ROOT/'_source_check_l057_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
