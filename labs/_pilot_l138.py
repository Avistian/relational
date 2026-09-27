"""Materialize the full source graph and time real training/evaluation batches."""
import json,time,hashlib,os
from pathlib import Path

def run(root):
 import torch,numpy as np
 from sentence_transformers import SentenceTransformer
 from relbench.datasets import get_dataset
 from relbench.tasks import get_task
 from relbench.modeling.utils import get_stype_proposal
 from torch_frame.config import TextEmbedderConfig
 from torch_geometric.loader import NeighborLoader
 from torch_geometric.seed import seed_everything
 from relkit.rdl_l117 import make_pkey_fkey_graph,get_node_train_table_input,Model
 root=Path(root);start=time.perf_counter();torch.set_num_threads(2);seed_everything(42)
 report=dict(status='RUNNING',phase='load_full_database',training='NOT_RUN')
 def save():
  report['seconds']=time.perf_counter()-start;(root/'training_pilot.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
 save()
 dataset=get_dataset('rel-amazon');dataset.cache_dir=str(root/'unpacked');db=dataset.get_db()
 task=get_task('rel-amazon','user-churn');task.cache_dir=str(root/'unpacked/user-churn')
 types=get_stype_proposal(db);report['rows']={k:len(v) for k,v in db.table_dict.items()};report['stypes']={k:{c:str(v) for c,v in d.items()} for k,d in types.items()};report['phase']='materialize';save()
 text=SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d',revision='e5e8fec6971be8960cfaa853a77a6ddc62a265d7',device='cuda')
 def embed(strings):return torch.from_numpy(text.encode(strings,show_progress_bar=False))
 data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256),cache_dir=str(root/'materialized'))
 del text,db;torch.cuda.empty_cache();report['phase']='graph_ready';report['preparation_seconds']=time.perf_counter()-start;save()
 torch.save((data,stats),root/'graph.pt')
 loaders={}
 for split in ['train','val','test']:
  q=get_node_train_table_input(task.get_table(split),task)
  loaders[split]=NeighborLoader(data,num_neighbors=[128,64],time_attr='time',input_nodes=q.nodes,input_time=q.time,transform=q.transform,batch_size=512,temporal_strategy='uniform',shuffle=split=='train',num_workers=0)
 model=Model(data,stats,2,128,1,'sum','batch_norm').cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.005)
 report['batches']={k:len(v) for k,v in loaders.items()};report['timings']={}
 for split in ['train','val']:
  train=split=='train';model.train(train);durations=[];clock=time.perf_counter()
  for i,batch in enumerate(loaders[split]):
   batch=batch.to('cuda')
   with torch.set_grad_enabled(train):
    pred=model(batch,task.entity_table).view(-1)
    if train:
     optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,batch[task.entity_table].y.float());loss.backward();optimizer.step()
   torch.cuda.synchronize();durations.append(time.perf_counter()-clock);clock=time.perf_counter()
   if i>=19:break
  report['timings'][split]=dict(batch_seconds=durations,mean_warm=float(np.mean(durations[2:])))
  report['phase']='timed_'+split;save()
 per_run=10*(min(report['batches']['train'],2001)*report['timings']['train']['mean_warm']+report['batches']['val']*report['timings']['val']['mean_warm'])+(report['batches']['val']+report['batches']['test'])*report['timings']['val']['mean_warm']
 report['projected_five_run_seconds']=5*per_run;report['gpu_peak_bytes']=torch.cuda.max_memory_allocated();report['status']='COMPLETE';report['training']='20 train batches only; not a score reproduction';save();return report
