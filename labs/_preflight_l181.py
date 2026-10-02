"""Execute the pinned baseline encoder's numerical path before any paid dispatch.

This is an actual component prerequisite, not a full sampled GNN update.
"""
import sys,json,hashlib,importlib.util,inspect
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l181';E=P/'evidence/l181';packet=E/'packet'
sys.path.insert(0,str(S/'upstream'))
import numpy as np,pandas as pd,torch,torch_frame
from relbench.base import Database,Table
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.nn import HeteroEncoder
from torch_frame.data.stats import StatType
torch.set_num_threads(1);config=json.loads((packet/'config.json').read_text());runs=[]
# Full source uses all dates in graph materialization. Numeric-only probes isolate
# the missing-value operation without introducing a substitute text encoder.
for task,spec in config['tasks'].items():
 tables={}
 for name,m in config['tables'].items():
  d=pd.read_parquet(packet/'db'/(name+'.parquet'))
  if name==spec['table']:d=d.drop(columns=['position',*spec['proxies']])
  tables[name]=Table(d,json.loads(m['fkey_col_to_pkey_table']),json.loads(m['pkey_col']),json.loads(m['time_col']))
 np.random.seed(0);torch.manual_seed(0);types=get_stype_proposal(Database(tables))
 for name,table in tables.items():
  keys=[table.pkey_col,*table.fkey_col_to_pkey_table]
  cols=sorted(c for c,t in types[name].items() if t==torch_frame.numerical and c not in keys)
  if not cols:continue
  full=table.df[cols].copy();ds=torch_frame.data.Dataset(full,{c:torch_frame.numerical for c in cols}).materialize()
  allowed=np.ones(len(full),bool) if table.time_col is None else (table.df[table.time_col]<pd.Timestamp('2005-01-01')).to_numpy()
  # Check the entire available numerical input population, no fitting/training loop.
  idx=np.flatnonzero(allowed);tf=ds.tensor_frame[torch.as_tensor(idx)]
  enc=HeteroEncoder(128,{name:tf.col_names_dict},{name:ds.col_stats});enc.eval()
  out=enc({name:tf})[name];loss=out.square().mean();loss.backward()
  bad={n:int((~torch.isfinite(p.grad)).sum()) for n,p in enc.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
  record=dict(task=task,table=name,columns=cols,rows=len(idx),full_fit_rows=len(full),future_fit_rows=int((~allowed).sum()),missing_cells={c:int(full.iloc[idx][c].isna().sum()) for c in cols},finite_forward=bool(torch.isfinite(out).all()),nonfinite_gradient_elements=sum(bad.values()),bad_parameters=bad)
  if bad:
   # Same values with missing input imputed BEFORE multiplication: diagnostic only.
   control=full.iloc[idx].copy();control=control.fillna(full.mean()).fillna(0)
   cds=torch_frame.data.Dataset(control,{c:torch_frame.numerical for c in cols}).materialize()
   torch.manual_seed(0);ce=HeteroEncoder(128,{name:cds.tensor_frame.col_names_dict},{name:cds.col_stats});ce.eval();co=ce({name:cds.tensor_frame})[name];co.square().mean().backward()
   record['early_imputation_control_nonfinite']=sum(int((~torch.isfinite(p.grad)).sum()) for p in ce.parameters() if p.grad is not None)
   full.iloc[idx].to_parquet(packet/(task+'-'+name+'-gradient-input.parquet'),index=False)
  runs.append(record)
report=dict(status='FAIL_NONFINITE_GRADIENT' if any(r['nonfinite_gradient_elements'] for r in runs) else 'PASS_NUMERICAL_ENCODER_PROBE',scope='Pinned HeteroEncoder numerical path on all pre2005 rows (untimed tables retained); no sampled GNN/optimizer execution',torch_version=torch.__version__,torch_frame_version=torch_frame.__version__,device='cpu',runs=runs,full_gnn_update='NOT_RUN',cloud_usd=0)
(packet/'gradient-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
for cls,n in [(torch_frame.nn.LinearEncoder,'LinearEncoder'),(torch_frame.nn.StypeEncoder,'StypeEncoder')]:
 (S/(n+'.py')).write_text(inspect.getsource(cls))
print(json.dumps(report,indent=2))
