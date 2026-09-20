"""Run in a clean pinned CPU environment; actual NN fit, no source-oracle dependencies."""
import json,platform,sys,importlib.metadata as metadata
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import *
torch.set_num_threads(2);m=json.loads((P/'_sources_l093.json').read_text())
graph=load_oag(P/'data/l093/graph_NN.pk',m['nn_data']['sha256'])
r,_=train_oag(graph,0,protocol_config('smoke'))
result={'status':'PASS','scope':'Clean CPU runtime, original NN bytes, complete one-update diagnostic','python':platform.python_version(),'versions':{k:metadata.version(k) for k in ['torch','numpy','pandas','dill']},'ndcg':r['test_ndcg'],'mrr':r['test_mrr'],'full_CS_run':'NOT_RUN','live_colab':'NOT_CHECKED'}
(P/'_portable_l093_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
