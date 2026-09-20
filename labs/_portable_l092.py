"""Run this with the isolated requirements-l092-runtime interpreter; smoke scope only."""
import importlib.util,importlib.metadata as md,json,platform,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
from han_l092 import load_acm,train_acm
from _verify_l092 import verify
import torch
torch.set_num_threads(2);checks=verify();m=json.loads((P/'_sources_l092.json').read_text());data=load_acm(P/'data/l092',m)
run,_=train_acm(data,seed=99,epochs=2)
result={'status':'PASS','scope':'portable pinned runtime: operator checks and two full-graph training updates; full schedule NOT_RUN in this interpreter','checks':checks,'diagnostic_accuracy':run['test_accuracy'],'environment':{'python':platform.python_version(),**{x:md.version(x) for x in ['numpy','scipy','torch','scikit-learn']}},'live_colab':'NOT_CHECKED'}
(P/'_portable_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
