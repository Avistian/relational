"""Execute after installing requirements-l090-runtime.txt in an isolated virtualenv."""
import json,platform,subprocess,sys
from pathlib import Path
import numpy as np,scipy,torch
from _verify_l090 import verify
R=Path(__file__).resolve().parent
assert sys.prefix!=sys.base_prefix
assert 'include-system-site-packages = false' in (Path(sys.prefix)/'pyvenv.cfg').read_text()
checks=verify()
subprocess.run([sys.executable,str(R/'_run_l090.py'),'--lane','inductive','--preset','paper','--output','_portable_inductive_l090_results.json'],check=True)
paper=json.loads((R/'_portable_paper_l090_results.json').read_text());ind=json.loads((R/'_portable_inductive_l090_results.json').read_text())
assert len(paper['runs'])==100 and len(ind['runs'])==3
r={'status':'PASS','fresh_venv':True,'system_site_packages':False,'python':platform.python_version(),'versions':{'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'checks':checks,'paper_runs':100,'paper_mean':paper['mean'],'paper_verdict':paper['course_verdict'],'inductive_runs':3,'inductive_mean':ind['mean'],'historical_parity':'INCOMPARABLE'}
r['exact_gcn_runs_match_author']=paper['runs']==json.loads((R/'_paper_l090_results.json').read_text())['runs']
r['exact_inductive_runs_match_author']=ind['runs']==json.loads((R/'_inductive_l090_results.json').read_text())['runs']
(R/'_clean_environment_l090_results.json').write_text(json.dumps(r,indent=2)+'\n')
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);(R/'requirements-l090-lock.txt').write_text(freeze)
print(r)
