"""Run in the newly installed CPU environment; fetch Cora into an empty directory."""
import json,tempfile,platform,hashlib
from pathlib import Path
import torch,numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from oversmoothing_l085 import karate_experiment,train_depth
from gcn_l082 import load_cora
LAB=Path(__file__).resolve().parent;torch.set_num_threads(1)
graph=json.loads((LAB/'sources/l085/karate.json').read_text())
actual=karate_experiment(graph,100);expected=json.loads((LAB/'_paper_l085_results.json').read_text())
assert actual['runs']==expected['runs']
with tempfile.TemporaryDirectory(prefix='l085-clean-data-') as tmp:
    root=Path(tmp);(root/'_sources_l078.json').write_bytes((LAB/'_sources_l078.json').read_bytes())
    data=load_cora(root);actual_depth=train_depth(data,2,0)
expected_depth=next(r for r in json.loads((LAB/'_depth_l085_results.json').read_text())['runs'] if r['depth']==2 and r['seed']==0)
assert actual_depth==expected_depth
report={'status':'PASS','python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'karate_experiments':500,'karate_exact_coordinates':True,'cora_fresh_download':True,'cora_depth':2,'cora_seed':0,'cora_epochs':actual_depth['epochs'],'exact_validation_trace_and_metrics':True,'implementation_sha256':hashlib.sha256((LAB/'relkit/oversmoothing_l085.py').read_bytes()).hexdigest(),'scope':'Fresh isolated CPU environment; live Colab not checked'}
(LAB/'_clean_environment_l085_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
