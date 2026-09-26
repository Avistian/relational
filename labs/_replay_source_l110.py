"""Re-evaluate a saved course checkpoint through unmodified released TGN/evaluator.
Usage: python labs/_replay_source_l102.py CHECKPOINT_FOLDER DATA_FOLDER DEVICE SEED
"""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'sources/l102'))
from model.tgn import TGN as Released
from utils.utils import get_neighbor_finder,RandEdgeSampler
from utils.data_processing import Data
from evaluation.evaluation import eval_edge_prediction
from relkit.checkpoint_l110 import load_wikipedia

def replay(folder,data_folder,device,seed):
 torch.set_num_threads(1);folder=Path(folder)
 nodes,edges,data,_=load_wikipedia(data_folder)
 def released_data(d):return Data(d['u'],d['v'],d['t'],d['e'],np.zeros(len(d['u'])))
 full=released_data(data['full']);finder=get_neighbor_finder(full,False)
 model=Released(finder,nodes,edges,device,n_layers=1,n_heads=2,dropout=.1,use_memory=True,
                memory_dimension=172,message_function='identity',aggregator_type='last',memory_updater_type='gru').to(device)
 saved=torch.load(folder/f'seed-{seed}.pt',map_location=device,weights_only=False)
 mapping={'time_encoder':'time_encoder','gru':'memory_updater.memory_updater','merge':'embedding_module.attention_models.0.merger','attention':'embedding_module.attention_models.0.multi_head_target','decoder':'affinity_score'}
 params=dict(model.named_parameters())
 with torch.no_grad():
  for name,value in saved['weights'].items():
   prefix,_,tail=name.partition('.')
   if prefix in mapping:params[mapping[prefix]+'.'+tail].copy_(value)
 memory,clock,pending=saved['temporal_state'];snapshot=(memory,clock,{k:[v] for k,v in pending.items()})
 predictions=np.load(folder/f'seed-{seed}-predictions.npz');results={}
 original=model.compute_edge_probabilities
 collected=[]
 def traced(*a,**kw):
  p,n=original(*a,**kw);collected.append((p.detach().cpu().numpy().flatten(),n.detach().cpu().numpy().flatten()));return p,n
 model.compute_edge_probabilities=traced
 for lane,key,sampler_seed in [('all','test',2),('new','new_test',3)]:
  model.memory.restore_memory(snapshot);collected.clear()
  pool=data['full'] if lane=='all' else data['new_test']
  sampler=RandEdgeSampler(pool['u'],pool['v'],seed=sampler_seed)
  ap,auc=eval_edge_prediction(model,sampler,released_data(data[key]),10)
  positive=np.concatenate([x[0] for x in collected]);negative=np.concatenate([x[1] for x in collected])
  err=max(float(np.max(np.abs(positive-predictions[lane+'_positive']))),float(np.max(np.abs(negative-predictions[lane+'_negative_score']))))
  assert err<2e-5,(lane,err)
  assert abs(ap-saved['result'][key]['ap'])<1e-5
  results[lane]={'events':len(positive),'max_probability_error':err,'released_ap':float(ap),'saved_ap':saved['result'][key]['ap'],'released_auc':float(auc)}
 report={'status':'PASS','seed':seed,'device':device,'source_commit':'e38cdf85998c6ca077167610dc4e769a688efa95','results':results,'scope':'Full selected seed evaluation from identical saved temporal state, using unmodified released model and evaluator'}
 (folder/f'seed-{seed}-source-replay.json').write_text(json.dumps(report,indent=2));print(report);return report
if __name__=='__main__':replay(*sys.argv[1:4],int(sys.argv[4]))
