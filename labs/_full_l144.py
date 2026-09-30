"""Visible full-data released-protocol runner; pilot never counts as a paper fit."""
import copy,hashlib,json,time,os,sys
from pathlib import Path
import numpy as np
import torch
from relkit.context_l144 import audit_cutoffs,keyed_map
from relkit.contextgnn_l144 import ContextGNN,ShallowRHSGNN,RHSEmbeddingMode

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()

def prepare(root,source):
 from relbench.datasets import get_dataset
 from relbench.tasks import get_task
 from relbench.modeling.utils import get_stype_proposal
 from relbench.modeling.graph import make_pkey_fkey_graph
 from torch_frame.config import TextEmbedderConfig
 from sentence_transformers import SentenceTransformer
 from torch_geometric.seed import seed_everything
 root=Path(root);root.mkdir(parents=True,exist_ok=True);seed_everything(42)
 dataset=get_dataset('rel-trial',download=True);task=get_task('rel-trial','site-sponsor-run',download=True)
 db=dataset.get_db();types=get_stype_proposal(db)
 tables={s:task.get_table(s,mask_input_cols=False) for s in ['train','val','test']}
 counts={s:len(t) for s,t in tables.items()};print('COUNTS',counts,flush=True)
 # Reconstruct source SQL on the full database before any training.
 import pandas as pd
 rebuilt={};full_db=dataset.get_db(upto_test_timestamp=False)
 for split,table in tables.items():
  check=task.filter_dangling_entities(task.make_table(full_db,pd.DatetimeIndex(sorted(table.df[task.time_col].unique()))))
  def keyed(t):return {(int(row[task.src_entity_col]),int(pd.Timestamp(row[task.time_col]).value)):sorted(map(int,row[task.dst_entity_col])) for _,row in t.df.iterrows()}
  assert keyed(check)==keyed(table),'Raw SQL label mismatch'
  rebuilt[split]=len(check)
 (root/'label-audit.json').write_text(json.dumps({'status':'PASS','source_SQL_reconstructed':rebuilt,'independent_algorithm':'NOT_CHECKED'},indent=2))
 text=SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d',revision='e5e8fec6971be8960cfaa853a77a6ddc62a265d7',device='cuda')
 def embed(strings):return torch.from_numpy(text.encode(strings,show_progress_bar=False))
 data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256),cache_dir=str(root/'materialized'))
 torch.save((data,stats),root/'graph.pt')
 meta={'graph_sha256':sha(root/'graph.pt'),'counts':counts,'rows':{k:len(t) for k,t in db.table_dict.items()},'eval_k':task.eval_k,'num_dst_nodes':task.num_dst_nodes,'src':task.src_entity_table,'dst':task.dst_entity_table,'stypes':types,'database_files':{str(p.relative_to(Path(dataset.cache_dir))):sha(p) for p in Path(dataset.cache_dir).rglob('*.parquet')},'task_files':{p.name:sha(p) for p in Path(task.cache_dir).rglob('*.parquet')}}
 (root/'prepared.json').write_text(json.dumps(meta,indent=2,default=str));return meta

def run_fit(root,output,arm,cfg,seed=42,epochs=20,trial=None,pilot=False):
 from relbench.datasets import get_dataset
 from relbench.tasks import get_task
 from relbench.modeling.graph import get_link_train_table_input
 from relbench.modeling.loader import SparseTensor
 from torch_geometric.loader import NeighborLoader
 from torch_geometric.seed import seed_everything
 from torch_geometric.utils.cross_entropy import sparse_cross_entropy
 torch.set_num_threads(1);seed_everything(seed);start=time.perf_counter();root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
 meta=json.loads((root/'prepared.json').read_text());assert sha(root/'graph.pt')==meta['graph_sha256']
 data,stats=torch.load(root/'graph.pt',weights_only=False);task=get_task('rel-trial','site-sponsor-run',download=False)
 tables={s:task.get_table(s,mask_input_cols=False) for s in ['train','val','test']};loaders={};inputs={}
 for s,t in tables.items():
  inp=get_link_train_table_input(t,task);inputs[s]=inp
  loaders[s]=NeighborLoader(data,num_neighbors=[128,64,32,16],time_attr='time',input_nodes=inp.src_nodes,input_time=inp.src_time,subgraph_type='bidirectional',batch_size=cfg['batch_size'],temporal_strategy='last',shuffle=s=='train',num_workers=0)
 cls=ContextGNN if arm=='contextgnn' else ShallowRHSGNN
 kwargs=dict(data=data,col_stats_dict=stats,num_layers=4,channels=cfg['channels'],embedding_dim=cfg['embedding_dim'],norm=cfg['norm'],rhs_emb_mode=RHSEmbeddingMode(cfg['rhs_emb_mode']),num_nodes=inputs['train'].num_dst_nodes,dst_entity_table=task.dst_entity_table,torch_frame_model_kwargs={'channels':cfg['encoder_channels'],'num_layers':cfg['encoder_layers']})
 model=cls(**kwargs).to('cuda');model.reset_parameters();opt=torch.optim.Adam(model.parameters(),lr=cfg['base_lr']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=cfg['gamma_rate'])
 target=SparseTensor(inputs['train'].dst_nodes[1],device='cuda');audit={'queries':0,'timed_nodes':0,'future_violations':0};source_batch=None
 def audit_batch(b):
  cutoff=b[task.src_entity_table].seed_time.numpy();audit['queries']+=len(cutoff)
  for typ,times in b.time_dict.items():audit['timed_nodes']+=audit_cutoffs(times.numpy(),b[typ].batch.numpy(),cutoff)
 @torch.no_grad()
 def evaluate(split):
  model.eval();pred=[];ids=[]
  for batch_index,b in enumerate(loaders[split]):
   audit_batch(b);b=b.to('cuda');logits=model(b,task.src_entity_table,task.dst_entity_table)
   assert torch.isfinite(logits).all(),'Nonfinite logits'
   pred.append(logits.sigmoid().topk(task.eval_k,dim=1).indices.cpu().numpy());ids.extend(b[task.src_entity_table].input_id.cpu().tolist())
   if pilot and batch_index>=3:break
  p=np.concatenate(pred);assert ids==list(range(len(ids)))
  if not pilot:assert len(ids)==len(tables[split])
  frame=tables[split].df.iloc[:len(p)];keys=list(zip(frame[task.src_entity_col].astype(int),frame[task.time_col].astype('int64')))
  value=keyed_map(keys,frame[task.dst_entity_col],keys,p,task.eval_k)
  from relbench.base import Table
  official=task.evaluate(p,Table(df=frame,fkey_col_to_pkey_table=tables[split].fkey_col_to_pkey_table,time_col=task.time_col))['link_prediction_map'];assert abs(value-official)<1e-12
  return value,p
 history=[];best=0.;selected=None;best_epoch=None;best_test=None;best_val_pred=None;nonfinite={}
 for epoch in range(1,epochs+1):
  clock=time.perf_counter();model.train();loss_sum=0.;n=0;steps=0
  for b in loaders['train']:
   audit_batch(b)
   if source_batch is None:source_batch=copy.deepcopy(b)
   b=b.to('cuda');src,dst=target[b[task.src_entity_table].input_id];opt.zero_grad();logits=model(b,task.src_entity_table,task.dst_entity_table)
   loss=sparse_cross_entropy(logits,torch.stack([src,dst]));assert torch.isfinite(loss),'Nonfinite loss';loss.backward()
   if epoch==1 and steps==0:nonfinite={name:int((~torch.isfinite(p.grad)).sum()) for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
   opt.step();bs=b[task.src_entity_table].batch_size;n+=bs;loss_sum+=float(loss.detach())*bs;steps+=1
   if steps%4==0:print('BATCH TIMING',arm,epoch,steps,time.perf_counter()-clock,flush=True)
   if (pilot and steps>=16) or steps>2000:break
  train_seconds=time.perf_counter()-clock;val_clock=time.perf_counter()
  val,vp=evaluate('val');validation_seconds=time.perf_counter()-val_clock
  if val>best:
   best=val;best_epoch=epoch;selected=copy.deepcopy(model.state_dict());best_val_pred=vp
   if not pilot:best_test=evaluate('test')
  scheduler.step();history.append({'epoch':epoch,'loss':loss_sum/n,'queries':n,'steps':steps,'val_map':val,'train_seconds':train_seconds,'validation_seconds':validation_seconds,'seconds':time.perf_counter()-clock})
  (out/'progress.json').write_text(json.dumps(history,indent=2));print(arm,history[-1],flush=True)
  if val<best-.001:break
  if trial is not None:
   trial.report(val,epoch)
   if trial.should_prune():
    import optuna
    raise optuna.TrialPruned()
 assert selected is not None,'No positive validation score'
 model.load_state_dict(selected);torch.save(selected,out/'selected.pt')
 # Real sampled forward-pass comparison with unmodified authors architecture.
 from contextgnn.nn.models import ContextGNN as Original,ShallowRHSGNN as OriginalShallow
 from contextgnn.utils import RHSEmbeddingMode as OriginalMode
 original_kwargs=dict(kwargs);original_kwargs['rhs_emb_mode']=OriginalMode(cfg['rhs_emb_mode'])
 ref=(Original if arm=='contextgnn' else OriginalShallow)(**original_kwargs).to('cuda');ref.load_state_dict(selected);ref.eval();model.eval()
 # Source evaluation caches RHS features across epochs. Clear only for this differential probe;
 # captured selection scores above preserve source behavior and its limitation.
 ref.rhs_embedding._cached_rhs_embedding=None;model.rhs_embedding._cached_rhs_embedding=None
 with torch.no_grad():
  b=source_batch.to('cuda');left=model(b,task.src_entity_table,task.dst_entity_table);right=ref(b,task.src_entity_table,task.dst_entity_table)
  delta=float((left-right).abs().max());assert torch.allclose(left,right,rtol=1e-5,atol=1e-5)
 preds={};split_values={'val':best}
 for split,p in [('val',best_val_pred)]+([] if pilot else [('test',best_test[1])]):
  frame=tables[split].df.iloc[:len(p)];targets=[list(map(int,v)) for v in frame[task.dst_entity_col]]
  preds[split+'_pred']=p;preds[split+'_entity']=frame[task.src_entity_col].to_numpy();preds[split+'_time']=frame[task.time_col].astype('int64').to_numpy()
  (out/(split+'-targets.json')).write_text(json.dumps(targets))
 if not pilot:split_values['test']=best_test[0]
 np.savez_compressed(out/'predictions.npz',**preds)
 result={'status':'PARTIAL_TIMING_PILOT' if pilot else 'COMPLETE','scope':{'full_query_counts':{s:len(t) for s,t in tables.items()},'training_batch_limit':16 if pilot else 2001,'validation_batch_limit':4 if pilot else None},'arm':arm,'cfg':cfg,'seed':seed,'history':history,'best_epoch':best_epoch,'scores':split_values,'seconds':time.perf_counter()-start,'temporal_audit':audit,'first_batch_nonfinite_gradients':nonfinite,'source_parity_max_abs':delta,'checkpoint_sha256':sha(out/'selected.pt'),'source_cache_behavior':'PRESERVED: RHS eval cache persists across training epochs'}
 (out/'result.json').write_text(json.dumps(result,indent=2));return result

PILOT_CONFIG=dict(channels=128,encoder_channels=128,encoder_layers=4,embedding_dim=64,norm='layer_norm',rhs_emb_mode='fusion',batch_size=256,base_lr=.001,gamma_rate=1.)

def full_search(root,output,arm):
 """Complete 50-trial plus five-repeat release search; no silent reduced preset."""
 import optuna
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 def objective(trial):
  cfg={key:trial.suggest_categorical(key,choices) for key,choices in {'encoder_channels':[32,64,128,256,512],'encoder_layers':[2,4,8],'channels':[32,64,128,256,512],'embedding_dim':[32,64,128,256,512],'norm':['layer_norm','batch_norm'],'rhs_emb_mode':['fusion','feature','lookup'],'batch_size':[256,512,1024]}.items()}
  cfg['base_lr']=trial.suggest_float('base_lr',.001,.01,log=True);cfg['gamma_rate']=trial.suggest_float('gamma_rate',.8,1.,log=True)
  try:r=run_fit(root,output/f'trial-{trial.number}',arm,cfg,42+trial.number,trial=trial)
  except RuntimeError as error:
   if 'out of memory' not in str(error).lower():raise
   (output/f'trial-{trial.number}'/'failure.txt').write_text(str(error));torch.cuda.empty_cache();return 0.
  return r['scores']['val']
 # Deterministic reconstruction: release left Optuna sampler unseeded and used a continuous RNG stream.
 study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.MedianPruner())
 study.optimize(objective,n_trials=50)
 study.trials_dataframe().to_json(output/'trials.json',orient='records')
 results=[run_fit(root,output/f'repeat-{s}',arm,study.best_params,100+s) for s in range(5)]
 summary={'status':'COMPLETE','best_config':study.best_params,'repeats':results,'aggregate':{split:{'mean':float(np.mean([r['scores'][split] for r in results])),'sample_sd':float(np.std([r['scores'][split] for r in results],ddof=1))} for split in ['val','test']},'paper_test_target':.2802 if arm=='contextgnn' else .1066,'descriptive_tolerance':.02,'rng_deviation':'Optuna seeded42; trial seed42+i; repeats100..104; original continuous/unseeded search history unavailable'}
 (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary
