"""Verify raw projection against independent pandas parsing and original reindex."""
import importlib.util,json,hashlib,sys,ast,types
from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('visible',P/'relkit/edgebank_l106.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources=json.loads((P/'_sources_l106.json').read_text())
for f in sources['files']:assert hashlib.sha256((P.parent/f['path']).read_bytes()).hexdigest()==f['sha256']
tree=ast.parse((P/'sources/l106/preprocess_data.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reindex');scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned original reindex>','exec'),scope);original=types.SimpleNamespace(**scope)
raw=P/'data/l102/wikipedia.csv';events=m.load_events(raw)
df=pd.read_csv(raw,header=None,skiprows=1,usecols=[0,1,2]);df.columns=['u','i','ts'];df['idx']=np.arange(len(df));projected=original.reindex(df)
np.testing.assert_array_equal(events,projected[['u','i','ts']].values)
r={'status':'PASS','rows':len(df),'projection':'Complete pandas projection + original bipartite reindex agrees with standalone parser','raw_sha256':m.DATA_SHA,'source_commit':sources['commit'],'source_hashes':'PASS','historical_bytes':'NOT_ESTABLISHED'}
(P/'_provenance_l106_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
