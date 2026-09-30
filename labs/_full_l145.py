"""Full selected released search plus explicitly partial timing pilots.

Source defects are preserved for forensic replay. Temporal failure blocks the
clean-reproduction gate. All nine schedules remain available for source replay.
"""
import copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,DistributedSampler
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relkit.relgt_l145 import RelGT
from relkit.relgt_contracts_l145 import audit_tokens,last_validation_min,keyed_mae

CONFIGS=[dict(layers=l,dropout=d) for d in [.3,.4,.5] for l in [1,4,8]]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def task_data(prior):
    from relbench.modeling.graph import get_node_train_table_input
    root=Path(prior);meta=json.loads((root/'prepared.json').read_text())
    assert sha(root/'graph.pt')==meta['graph_sha256']
    for n,h in meta['archives'].items():assert sha(root/'cache'/n)==h
    data,stats=torch.load(root/'graph.pt',weights_only=False)
    ds=get_dataset('rel-f1');ds.cache_dir=str(root/'unpacked');ds.get_db.cache_clear()
    task=get_task('rel-f1','driver-position');task.cache_dir=str(root/'unpacked/driver-position');task.get_table.cache_clear()
    return ds,task,data,stats,meta

def prepare(root,prior,source_root):
    import sys,h5py
    sys.path.insert(0,str(source_root));import utils as source
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ds,task,data,stats,meta=task_data(prior)
    # Identical per-query operation and last-entity overwrite, evaluated serially.
    # Each query has its own seed, so only the last repeated entity needs computation.
    def serial(data,K,table_input_nodes,table_input_time,undirected=True,num_workers=None):
        adjacency=source.build_adjacency_hetero(data,undirected)
        all_nodes=[(t,i) for t in data.node_types for i in range(data[t].num_nodes)]
        source.init_worker_globals(adjacency,all_nodes)
        nt,ids=table_input_nodes;latest={}
        for idx,stamp in zip(ids.tolist(),table_input_time.tolist()):latest[idx]=stamp
        result={nt:{}}
        for idx,stamp in latest.items():
            seed=hash((nt,idx,stamp,K))&0xffffffff
            _,_,tokens,edges=source._process_one_seed((data,K,nt,idx,stamp,seed))
            result[nt][idx]=(tokens,edges)
        return result
    source.local_nodes_hetero=serial
    reports={};sample_rows=[];label_count=0
    # Full future-inclusive raw DB is used ONLY to reconstruct labels.
    raw=ds.get_db(upto_test_timestamp=False).table_dict['results'].df
    for split in ['train','val','test']:
        cache=source.RelGTTokens(data,task,300,split=split,num_workers=1,precomputed_dir=str(root/'tokens'))
        table=task.get_table(split,mask_input_cols=False).df
        keys=list(zip(table[task.entity_col].astype(int),table[task.time_col].astype('int64')//10**9))
        labels=[]
        import pandas as pd
        for entity,stamp in keys:
            rows=raw[(raw.driverId==entity)&(raw.date>pd.Timestamp(stamp,unit='s'))&(raw.date<=pd.Timestamp(stamp,unit='s')+pd.Timedelta(days=60))]
            labels.append(rows.positionOrder.mean())
        np.testing.assert_allclose(labels,table[task.target_col],atol=1e-7,rtol=1e-7);label_count+=len(labels)
        with h5py.File(cache.precomputed_path) as hf:
            types=hf['types'][:];ids=hf['indices'][:];times=hf['times'][:]
            token_times=[];overwritten=0;examples=[]
            last=dict(zip(table[task.entity_col].astype(int),keys))
            for i,key in enumerate(keys):
                stamps=[]
                for j,(t,n) in enumerate(zip(types[i],ids[i])):
                    name=cache.index_to_node_type[int(t)]
                    stamps.append(None if 'time' not in data[name] else int(data[name].time[int(n)]))
                token_times.append(stamps)
                if key!=last[key[0]]:overwritten+=1
            violations=audit_tokens(keys,token_times)
            for i,j in violations[:12]:examples.append(dict(row=i,key=keys[i],slot=j,type=cache.index_to_node_type[int(types[i,j])],entity=int(ids[i,j]),actual_time=token_times[i][j],cached_age=float(times[i,j])))
            reports[split]=dict(queries=len(keys),unique_entities=len(set(k[0] for k in keys)),overwritten_queries=overwritten,future_token_occurrences=len(violations),affected_queries=len(set(i for i,j in violations)),examples=examples,cache_sha256=sha(cache.precomputed_path))
            np.savez_compressed(root/f'{split}-audit.npz',entity=np.array([k[0] for k in keys]),cutoff=np.array([k[1] for k in keys]),target=np.array(labels),token_times=np.array([[(-1 if x is None else x) for x in row] for row in token_times]),types=types,indices=ids,ages=times)
    report=dict(graph=meta,graph_provenance='REUSED hash-verified L143 materialization; not fresh preprocessing',split_audits=reports,independently_rebuilt_labels=label_count,source_revision='19e423ca3e7cac761130aba790857f2dc3a46ef7',pythonhashseed=os.environ.get('PYTHONHASHSEED'),temporal_status='FAIL' if any(r['future_token_occurrences'] for r in reports.values()) else 'PASS')
    (root/'audit.json').write_text(json.dumps(report,indent=2));return report

def run_fit(root,prior,source_root,output,layers=1,dropout=.3,pilot=False):
    import sys
    sys.path.insert(0,str(source_root));import utils as source
    from torch_geometric.seed import seed_everything
    seed_everything(0);torch.set_num_threads(1);start=time.perf_counter()
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ds,task,data,stats,meta=task_data(prior)
    caches={s:source.RelGTTokens(data,task,300,split=s,num_workers=1,precomputed_dir=str(root/'tokens')) for s in ['train','val','test']}
    cache_audit=json.loads((root/'audit.json').read_text())
    for split,cache in caches.items():
        assert sha(cache.precomputed_path)==cache_audit['split_audits'][split]['cache_sha256'],'Token cache changed after audit'
    samplers={s:DistributedSampler(c,num_replicas=1,rank=0,shuffle=s=='train',seed=0) for s,c in caches.items()}
    loaders={s:DataLoader(c,batch_size=256,sampler=samplers[s],collate_fn=c.collate,num_workers=0) for s,c in caches.items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    kwargs=dict(num_nodes=data.num_nodes,max_neighbor_hop=3,node_type_map=caches['train'].node_type_to_index,col_names_dict={t:data[t].tf.col_names_dict for t in data.node_types},col_stats_dict=stats,local_num_layers=layers,channels=512,out_channels=1,global_dim=256,heads=4,ff_dropout=dropout,attn_dropout=dropout,conv_type='full',num_centroids=4096,sample_node_len=300)
    model=RelGT(**kwargs).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    clamp=np.percentile(task.get_table('train').df[task.target_col],[2,98]);history=[];best=None;best_score=float('inf');parity=None;grad=None
    def forward(batch,m):
        grouped={k:batch[k] for k in ['grouped_tfs','grouped_indices','flat_batch_idx','flat_nbr_idx']}
        return m(*[batch[k].to(device) for k in ['neighbor_types','node_indices','neighbor_hops','neighbor_times']],grouped,edge_index=batch['edge_index'].to(device),batch=batch['batch'].to(device)).view(-1)
    def evaluate(split,limit=None):
        model.eval();pred=[];order=[]
        with torch.no_grad():
            for i,b in enumerate(loaders[split]):
                if limit is not None and i>=limit:break
                pred.extend(forward(b,model).clamp(*clamp).cpu().tolist());order.extend(b['global_idx'].tolist())
        tab=task.get_table(split,mask_input_cols=False).df.iloc[order]
        keys=list(zip(tab[task.entity_col].astype(int),tab[task.time_col].astype('int64')//10**9));y=tab[task.target_col].to_numpy()
        return keyed_mae(keys,y,keys,pred),dict(entity=np.array([k[0] for k in keys]),cutoff=np.array([k[1] for k in keys]),target=y,pred=np.array(pred))
    for epoch in range(1,(1 if pilot else 100)+1):
        ep=time.perf_counter();samplers['train'].set_epoch(epoch);model.train();queries=0;steps=0
        for step,b in enumerate(loaders['train'],1):
            if pilot and step>3:break
            if step>3000:break
            optimizer.zero_grad();p=forward(b,model);loss=torch.nn.functional.l1_loss(p,b['labels'].to(device));loss.backward()
            if grad is None:grad={n:int((~torch.isfinite(v.grad)).sum()) for n,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()}
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();queries+=len(p);steps+=1
        train_seconds=time.perf_counter()-ep;ev=time.perf_counter();val,packet=evaluate('val',1 if pilot else None);val_seconds=time.perf_counter()-ev
        history.append(dict(epoch=epoch,queries=queries,steps=steps,train_seconds=train_seconds,val_seconds=val_seconds,val_mae=val))
        if last_validation_min([h['val_mae'] for h in history])==len(history)-1:best_score=val;best=copy.deepcopy(model.state_dict())
        if not pilot:(out/'history.json').write_text(json.dumps(history))
    model.load_state_dict(best);torch.save(best,out/'selected.pt')
    scores={'val':best_score};predictions={'val_'+k:v for k,v in packet.items()}
    if not pilot:
        scores['val'],packet=evaluate('val');predictions={'val_'+k:v for k,v in packet.items()}
        scores['test'],packet=evaluate('test');predictions.update({'test_'+k:v for k,v in packet.items()})
    # Real-batch original model parity: align stochastic PE and dropout draws.
    from _mechanism_l145 import original_modules
    with original_modules(source_root) as Original:
        orig=Original(**kwargs).to(device);orig.load_state_dict(model.state_dict());orig.eval();model.eval()
        b=next(iter(loaders['val']))
        with torch.no_grad():
            torch.manual_seed(145);a=forward(copy.deepcopy(b),model)
            torch.manual_seed(145);z=forward(copy.deepcopy(b),orig)
        torch.testing.assert_close(a,z,atol=1e-5,rtol=1e-5);parity=float((a-z).abs().max())
    np.savez_compressed(out/'predictions.npz',**predictions)
    result=dict(status='PARTIAL_TIMING_PILOT' if pilot else 'COMPLETE_SOURCE_REPLAY',layers=layers,dropout=dropout,seed=0,epochs=len(history),history=history,scores=scores,test='NOT_RUN' if pilot else 'SOURCE_REPLAY_ONLY',best_epoch=last_validation_min([h['val_mae'] for h in history])+1,first_batch_nonfinite_gradients=grad,real_batch_original_max_error=parity,seconds=time.perf_counter()-start,checkpoint_sha256=sha(out/'selected.pt'),parameter_count=sum(p.numel() for p in model.parameters()),scope='Released token-cache defects preserved; not a temporally valid benchmark reproduction')
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

def full_search(root,prior,source_root,output,allow_invalid_source_replay=False):
    audit=json.loads((Path(root)/'audit.json').read_text())
    if audit['temporal_status']!='PASS' and not allow_invalid_source_replay:
        raise RuntimeError('Temporal audit failed: clean reproduction blocked; source replay requires explicit forensic mode')
    results=[]
    for c in CONFIGS:results.append(run_fit(root,prior,source_root,Path(output)/f"L{c['layers']}-D{c['dropout']}",**c))
    # Published search aggregation script was not released; this rule is declared.
    best=min(results,key=lambda r:r['scores']['val'])
    return dict(results=results,validation_selected=best,selection_rule='minimum final validation MAE; first config breaks exact ties')
