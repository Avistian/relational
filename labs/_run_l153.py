"""Pinned released BPR script with passive audits; pilot is never a paper fit."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from relkit.recommendation_model_l153 import Model
from relkit.recommendation_l153 import ranking_metrics,negative_audit
from relkit.batch_audit_l123 import audit_batch
from _prepare_l153 import sha
P=Path(__file__).resolve().parent

def instrument(source,pilot):
    replacements=[
      ('from model import Model',''),
      ('from tqdm import tqdm','def tqdm(x, **kwargs): return x'),
      ('text_embedder=GloveTextEmbedding(device=device)','text_embedder=None'),
      ('        x_src = model(src_batch, task.src_entity_table)','        observe_batch(src_batch, batch_pos_dst, batch_neg_dst)\n        x_src = model(src_batch, task.src_entity_table)'),
      ('        loss.backward()','        loss.backward()\n        observe_gradients(model)'),
      ('        if steps > args.max_steps_per_epoch:','        if steps >= 32:' if pilot else '        if steps > args.max_steps_per_epoch:'),
      ('        emb = model(batch, task.dst_entity_table).detach()','        observe_eval(batch, task.dst_entity_table)\n        emb = model(batch, task.dst_entity_table).detach()'),
      ('        emb = model(batch, task.src_entity_table)','        observe_eval(batch, task.src_entity_table)\n        emb = model(batch, task.src_entity_table)'),
      ('best_val_metric = 0','best_val_metric = 0\nselected_epoch = None\ntrace = []'),
      ('    train_loss = train()','    epoch_start = time.perf_counter()\n    train_loss = train()\n    train_seconds = time.perf_counter()-epoch_start\n    val_start = time.perf_counter()'),
      ('            state_dict = copy.deepcopy(model.state_dict())','            state_dict = copy.deepcopy(model.state_dict())\n            selected_epoch = epoch'),
      ('\n\nmodel.load_state_dict(state_dict)','\n    trace.append(dict(epoch=epoch,train_loss=train_loss,val_map=val_metrics[tune_metric],train_seconds=train_seconds,validation_seconds=time.perf_counter()-val_start,seconds=time.perf_counter()-epoch_start))\n    (output/"progress.json").write_text(json.dumps(trace,indent=2))\n\nmodel.load_state_dict(state_dict)'),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    if pilot:source=source.split('test_pred = test(')[0]
    return source

def run_fit(root,output,seed=0,pilot=False,model_class=Model,metric_fn=ranking_metrics,negative_fn=negative_audit):
    import pandas as pd
    import relbench.modeling.graph as graph
    import relbench.modeling.loader as loader
    import relbench.modeling.nn as nn
    from relbench.tasks import get_task
    from torch_geometric.seed import seed_everything
    torch.set_num_threads(1);start=time.perf_counter();root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((root/'prepared.json').read_text());assert sha(root/'graph.pt')==meta['graph_sha256'];assert json.loads((root/'independent-labels.json').read_text())['status']=='PASS'
    for name,module in [('graph.py',graph),('loader.py',loader),('nn.py',nn)]:assert Path(module.__file__).read_bytes()==(P/'sources/l153'/name).read_bytes(),name
    data,stats=torch.load(root/'graph.pt',weights_only=False)
    task=get_task('rel-trial','site-sponsor-run',download=False)
    tables={s:task.get_table(s,mask_input_cols=False).df for s in ['train','val','test']}
    # A training source can recur at different cutoffs: exact pair keys are essential.
    train_truth={(int(e),int(pd.Timestamp(t).timestamp())):set(map(int,p)) for e,t,p in tables['train'][[task.src_entity_col,task.time_col,task.dst_entity_col]].itertuples(index=False,name=None)}
    audit=dict(batches=0,queries=0,dated_node_occurrences=0,edge_occurrences=0,temporal_violations=0,train_batches=0,negative_comparisons=0,positive_collisions=0,negative_audit_batches=0,availability='NOT_OBSERVED')
    sample=None;first_grad=None
    def observe_eval(batch,entity):
        # CPU copies preserve RNG and original source computation.
        cpu=batch.cpu();r=audit_batch(cpu,data,entity)
        for k,v in r.items():audit[k]+=v
        batch.to('cuda' if torch.cuda.is_available() else 'cpu')
    def observe_batch(src,pos,neg):
        nonlocal sample
        for b,entity in [(src,task.src_entity_table),(pos,task.dst_entity_table),(neg,task.dst_entity_table)]:observe_eval(b,entity)
        n=src[task.src_entity_table].batch_size
        assert torch.equal(src[task.src_entity_table].seed_time,pos[task.dst_entity_table].seed_time)
        assert torch.equal(src[task.src_entity_table].seed_time,neg[task.dst_entity_table].seed_time)
        audit['train_batches']+=1
        if sample is None:sample=copy.deepcopy(src).cpu()
        # Predeclared first32batch diagnostic; full temporal audit covers every batch.
        if audit['negative_audit_batches']<32:
            ids=src[task.src_entity_table].n_id[:n].cpu().tolist();times=src[task.src_entity_table].seed_time.cpu().tolist();negids=neg[task.dst_entity_table].n_id[:n].cpu().tolist()
            positives=[train_truth[e,t] for e,t in zip(ids,times)]
            r=negative_fn(times,negids,positives,task.num_dst_nodes)
            audit['negative_comparisons']+=r['comparisons'];audit['positive_collisions']+=r['positive_collisions'];audit['negative_audit_batches']+=1
    def observe_gradients(model):
        nonlocal first_grad
        if first_grad is None:first_grad={name:int((~torch.isfinite(p.grad)).sum()) for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
    original=(P/'sources/l153/gnn_link.py').read_text();source=instrument(original,pilot);(output/'executed.py').write_text(source)
    cache=output/'cache';folder=cache/'rel-trial';folder.mkdir(parents=True);(folder/'stypes.json').write_bytes((root/'stypes.json').read_bytes())
    old=graph.make_pkey_fkey_graph;argv=sys.argv[:];graph.make_pkey_fkey_graph=lambda *a,**k:(data,stats)
    sys.path.insert(0,str(P/'sources/l153'));sys.argv=['gnn_link.py','--dataset','rel-trial','--task','site-sponsor-run','--seed',str(seed),'--epochs',str(1 if pilot else 20),'--cache_dir',str(cache)]
    env=dict(__name__='l153_fit',Model=model_class,time=time,output=output,observe_batch=observe_batch,observe_eval=observe_eval,observe_gradients=observe_gradients)
    try:exec(compile(source,'l153_pinned_gnn_link.py','exec'),env)
    finally:graph.make_pkey_fkey_graph=old;sys.argv=argv
    result=dict(status='PARTIAL_TIMING_PILOT' if pilot else 'COMPLETE',seed=seed,protocol_hash=sha(P/'sources/l153/protocol.json'),selected_epoch=env['selected_epoch'],trace=env['trace'],scores={},counts={},temporal_violations=0,audit=audit,first_batch_nonfinite_gradients=first_grad,source_sha256=hashlib.sha256(original.encode()).hexdigest(),executed_sha256=sha(output/'executed.py'),loader_batches=len(env['train_loader']),training_queries_per_full_epoch=len(env['train_loader'])*512)
    arrays={}
    for split in ['val'] if pilot else ['val','test']:
        df=tables[split];pred=env[split+'_pred'];truth=[dict(entity=int(e),time=int(pd.Timestamp(t).value),positives=list(map(int,p))) for e,t,p in df[[task.src_entity_col,task.time_col,task.dst_entity_col]].itertuples(index=False,name=None)]
        ranked=[dict(entity=t['entity'],time=t['time'],ranking=list(map(int,p))) for t,p in zip(truth,pred)]
        metrics=metric_fn(truth,list(reversed(ranked)),task.eval_k,task.num_dst_nodes);official=float(env[split+'_metrics']['link_prediction_map']);assert abs(metrics['map']-official)<1e-12
        result['scores'][split]=metrics['map'];result['counts'][split]=metrics['n'];result[split+'_metrics']=metrics
        arrays[split+'_pred']=pred;arrays[split+'_entity']=df[task.src_entity_col].to_numpy();arrays[split+'_time']=df[task.time_col].astype('int64').to_numpy()
    np.savez_compressed(output/'predictions.npz',**arrays)
    model=env['model'];torch.save(model.state_dict(),output/'selected.pt');result['checkpoint_sha256']=sha(output/'selected.pt')
    # Differential real-batch forward and input-parameter gradient comparison after fitting.
    import importlib.util
    spec=importlib.util.spec_from_file_location('l153_original_model',P/'sources/l153/model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ref=module.Model(data=data,col_stats_dict=stats,num_layers=2,channels=128,out_channels=128,aggr='sum',norm='layer_norm',shallow_list=[task.dst_entity_table]).to(env['device']);ref.load_state_dict(model.state_dict());ref.eval();model.eval()
    b=sample.to(env['device'])
    with torch.no_grad():
        left=model(b,task.src_entity_table);right=ref(b,task.src_entity_table);assert torch.isfinite(left).all();delta=float((left-right).abs().max());assert torch.allclose(left,right,rtol=1e-5,atol=1e-5)
    result['original_model_max_abs']=delta;result['seconds']=time.perf_counter()-start
    (output/'result.json').write_text(json.dumps(result,indent=2));return result
