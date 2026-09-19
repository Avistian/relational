"""Fresh runtime and fresh Cora download; compare canonical seed0 exactly."""
import json,tempfile,sys,hashlib,platform
from pathlib import Path
import torch,torch_geometric
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from gcn_l082 import load_cora
from pyg_l086 import as_data,train_citation,loader_check
LAB=Path(__file__).resolve().parent;torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='l086-fresh-data-') as tmp:
 root=Path(tmp);(root/'_sources_l078.json').write_bytes((LAB/'_sources_l078.json').read_bytes())
 data=as_data(load_cora(root));actual=train_citation(data,0)
expected=json.loads((LAB/'_paper_l086_results.json').read_text())['runs'][0]
assert actual==expected
sample=loader_check()
r={'status':'PASS','python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__,'fresh_environment':sys.prefix,'fresh_data_download':True,'exact_seed0_trace':True,'real_neighbor_loader':sample,'pyg_lib_wheel_sha256':'96fde7744dc81990b1af83ad74546a27ad016797ffc66c32039ca347b0445b34','pyg_lib_source':'edc9e2a88d1c5d0953b5f69c98b8365597c6b699','boundary':'Fresh environment reuses the locally built pinned-source wheel; no independent compiler build; live Colab NOT_CHECKED'}
(LAB/'_clean_environment_l086_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
