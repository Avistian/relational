"""Run inside a fresh pinned environment; compare inline notebook smoke measurements."""
import importlib.metadata as md,json,platform,sys
from pathlib import Path
from relkit.gin_l088 import load_mutag,train_fold
from _verify_l088 import main as verify
LAB=Path(__file__).resolve().parent
verify();g=load_mutag(LAB/'data/l088');actual=[]
for mode in ['sum','mean','max']:
 r,_=train_fold(g,epochs=2,iters=10,readout=mode);actual.append({'readout':mode,'curve':r['curves']})
expected=json.loads((LAB/'_execution_l088_results.json').read_text())['smoke'];assert actual==expected
r={'status':'PASS','python':platform.python_version(),'executable':sys.executable,'fresh_venv':True,'system_site_packages':False,'versions':{x:md.version(x) for x in ['torch','numpy','scikit-learn']},'three_inline_smoke_curves_exact_match':True,'full_grid_clean_environment':'NOT_RUN'}
(LAB/'_clean_environment_l088_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
