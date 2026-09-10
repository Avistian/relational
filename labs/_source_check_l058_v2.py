"""Independent check of upstream complete-panel average ranks, not training parity."""
import ast,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from relkit.talent_audit_l058 import ARMS,load_tables,rank_matrix
ROOT=Path(__file__).resolve().parent
path=ROOT/'sources/l058/audit/average_rank.py';source=path.read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='calculate_ranks_for_each_dataset')
ns={'pd':pd,'print':lambda *args:None};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
tables,_,_=load_tables(ROOT);checks=[]
for task,frame in tables.items():
 x=frame[ARMS]
 # Upstream flips numeric values then ranks descending: feed losses for all tasks.
 oriented=-x if task!='regression' else x
 # pandas 3 renamed applymap to map; preserve the upstream elementwise operation.
 with patch.object(pd,'read_excel',return_value=oriented.copy()), patch.object(pd.DataFrame,'applymap',pd.DataFrame.map,create=True):
  upstream=ns['calculate_ranks_for_each_dataset']('provided-complete-dataframe')
 local=rank_matrix(x.to_numpy(),task!='regression')
 np.testing.assert_array_equal(local,upstream.to_numpy())
 checks.append({'task':task,'rows':len(x),'columns':6,'max_delta':float(np.max(np.abs(local-upstream.to_numpy())))})
result={'status':'PASS','upstream_revision':'1b973adffa4203c3f62c4949957b1ba3efbe60d3',
 'upstream_function':'visualization/average_rank.py:calculate_ranks_for_each_dataset','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'checks':checks,'scope':'Exact complete-panel rank operator agreement after explicit metric orientation; pandas IO supplied in memory; no training/selector/Table7 parity',
 'compatibility':'pandas 3 DataFrame.applymap alias supplied as DataFrame.map; identical elementwise callback',
 'missingness':'Upstream mean-imputes missing ranks; local uses common complete panels. No imputation in the compared six-method rows.'}
(ROOT/'_source_check_l058_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
