"""Real-data training-health gate for the released RelGNN numerical encoder.

A prerequisite probe, not a complete graph fit or a benchmark score.
"""
import hashlib,importlib.util,json,sys,time,os
from pathlib import Path
import numpy as np,pandas as pd,torch,torch_frame
from torch_frame.data.stats import StatType
P=Path(__file__).resolve().parent;E=P/'evidence/l178';torch.set_num_threads(1);torch.manual_seed(0)
O=Path(os.environ.get('L178_DIAGNOSTIC_OUT',str(E)));O.mkdir(parents=True,exist_ok=True)
if (O/'gradient-preflight.json').exists():raise FileExistsError('Choose a fresh L178_DIAGNOSTIC_OUT directory; preserve original evidence')
# The original RelGNN source class is loaded verbatim, not the teaching rewrite.
from _parity_l141 import original_modules
with original_modules(P/'sources/l141') as _:HeteroEncoder=sys.modules['relgnn_nn'].HeteroEncoder
raw=pd.read_parquet(P/'evidence/l171/db/results.parquet')
query=json.loads((E/'fastdfs-preflight.json').read_text())['query'];cut=pd.Timestamp(query['cutoff']);driver=query['driverId']
train=np.load(E/'released-task/train.npz');idx=np.load(E/'support-schedule.npz')['indices'][0]
assert [driver,cut.value] in np.column_stack([train['driverId'][idx],train['date'][idx]]).tolist()
frame=raw[(raw.driverId==driver)&(raw.date<cut)].copy()
# Original type inference is deterministic under seed 42; only numerical columns
# enter this focused numerical-path probe. No target or future cell is used.
from relbench.base import Database,Table
from relbench.modeling.utils import get_stype_proposal
np.random.seed(42);torch.manual_seed(42)
table=Table(frame,{'raceId':'races','driverId':'drivers','constructorId':'constructors'},'resultId','date')
types=get_stype_proposal(Database({'results':table}))['results']
columns=sorted(c for c,t in types.items() if t==torch_frame.numerical and c not in ['resultId','raceId','driverId','constructorId'])
assert columns and len(frame)>1
materialized=torch_frame.data.Dataset(frame[columns],{c:torch_frame.numerical for c in columns}).materialize()
tf=materialized.tensor_frame
encoder=HeteroEncoder(channels=128,node_to_col_names_dict={'results':tf.col_names_dict},node_to_col_stats={'results':materialized.col_stats})
encoder.eval();out=encoder({'results':tf})['results']
assert torch.isfinite(out).all(),'Nonfinite encoder forward'
# A finite scalar readout drives every output coordinate; isolate the numerical
# encoder before graph sampling, target choice, optimizer or model selection.
loss=out.square().mean();loss.backward()
grad={n:dict(elements=p.grad.numel(),nonfinite=int((~torch.isfinite(p.grad)).sum())) for n,p in encoder.named_parameters() if p.grad is not None}
count=sum(x['nonfinite'] for x in grad.values())
report=dict(status='FAIL_NONFINITE_GRADIENT' if count else 'PASS',query=query,history_rows=len(frame),numerical_columns=columns,missing_cells={c:int(frame[c].isna().sum()) for c in columns},finite_forward=True,loss=float(loss.detach()),nonfinite_gradient_elements=count,gradients=grad,
    source_sha256=hashlib.sha256((P/'sources/l141/examples__relgnn_nn.py').read_bytes()).hexdigest(),torch_version=torch.__version__,torch_frame_version=torch_frame.__version__,device='cpu',scope='Released HeteroEncoder numerical branch on one actual support query history; no GNN training or model ranking')
(O/'gradient-preflight.json').write_text(json.dumps(report,indent=2)+'\n');frame[columns+['resultId','driverId','date']].to_parquet(O/'gradient-input.parquet')
print(json.dumps({k:v for k,v in report.items() if k!='gradients'},indent=2))
