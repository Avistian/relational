"""Execute complete lesson lanes and compare each baseline to original source."""
import ast,hashlib,json,platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import sklearn,pyarrow
from relkit.temporal_l101 import paper_replay,course_experiment,load_task,baseline_predict,ARMS
from _check_l101 import check
P=Path(__file__).resolve().parent
r=paper_replay(P/'data/l101',P/'_predictions_l101.npz')
c=course_experiment();checks=check()
# Execute the unmodified released evaluate() with a prediction-capture evaluator.
source=(P/'sources/l101/baseline_node.py').read_text()
fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
class Table:
 def __init__(self,df):self.df=df;self.fkey_col_to_pkey_table={'driverId':'drivers'}
 def __len__(self):return len(self.df)
ns={'np':np,'pd':pd,'Table':Table,'Dict':dict,'task':SimpleNamespace(target_col='position',evaluate=lambda pred,*args:pred)}
exec(ast.get_source_segment(source,fn),ns)
tables=load_task(P/'data/l101')
for split in ['val','test']:
 fit=tables['train'] if split=='val' else pd.concat([tables['train'],tables['val']])
 query=tables[split][['driverId','date']]
 for arm in ARMS:
  np.testing.assert_allclose(baseline_predict(fit,query,arm),ns['evaluate'](Table(fit),Table(query),arm),rtol=0,atol=1e-12)
assert all(x['verdict']=='MATCH' for x in r['results'])
assert all(x['test_accuracy']==1 for x in c['records'] if x['arm']!='legal')
for name,value in [('_paper_l101_results.json',r),('_experiment_l101_results.json',c),('_check_l101_results.json',checks)]:
 (P/name).write_text(json.dumps(value,indent=2)+'\n')
e={'status':'PASS','command':'.venv/bin/python labs/_verify_l101.py','python':platform.python_version(),
   'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'pyarrow':pyarrow.__version__},
   'source_prediction_parity':'PASS: all 10 vectors, absolute tolerance 1e-12','paper_target_cells':10,'matching_cells':10,
   'hashes':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'relkit/temporal_l101.py',P/'_predictions_l101.npz',*sorted((P/'sources/l101').glob('*.py'))]},
   'full_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_verify_l101_results.json').write_text(json.dumps(e,indent=2)+'\n')
print(json.dumps({'paper':[(x['arm'],x['split'],x['mae'],x['verdict']) for x in r['results']], 'teaching':[(x['seed'],x['arm'],x['test_accuracy']) for x in c['records']], 'checks':checks},indent=2))
