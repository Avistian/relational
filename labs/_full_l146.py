"""Full-data course comparison; not the historical RelGT/RDL protocol.

Source-faithful full RelGT search is retained separately in _full_l145.py.
"""
import copy,hashlib,json,random,time,uuid
from pathlib import Path
import numpy as np
import torch
from relkit.comparison_l146 import temporal_context,select_config,paired_errors
from relkit.gnn_l146 import CourseGNN
from relkit.relgt_course_l146 import RelGT
from _full_l145 import task_data,sha

CONFIG=dict(k=32,width=64,layers=2,dropout=.1,centroids=128,global_dim=32,batch=128,
            epochs=10,lr=.001,weight_decay=1e-5,seeds=[0,1,2],sampler_seed=146)

def prepare(root,prior):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    ds,task,data,stats,meta=task_data(prior)
    names=data.node_types;offsets=np.cumsum([0]+[data[t].num_nodes for t in names]);nt=len(names)
    node_type=np.repeat(np.arange(nt),np.diff(offsets));local_id=np.concatenate([np.arange(data[t].num_nodes) for t in names])
    times={int(i):None for i in range(int(offsets[-1]))}
    for t,off in zip(names,offsets):
        if 'time' in data[t]:
            for j,v in enumerate(data[t].time.tolist()):times[int(off+j)]=int(v)
    adjacency={i:set() for i in times};links={i:[] for i in times}
    for rid,(src,rel,dst) in enumerate(data.edge_types):
        s=names.index(src);d=names.index(dst)
        for a,b in data[(src,rel,dst)].edge_index.t().tolist():
            a+=int(offsets[s]);b+=int(offsets[d]);adjacency[a].add(b);adjacency[b].add(a);links[a].append((b,rid))
    import pandas as pd
    raw=ds.get_db(upto_test_timestamp=False).table_dict['results'].df
    summaries={};total_labels=0
    for split in ['train','val','test']:
        tab=task.get_table(split,mask_input_cols=False).df
        entity=tab[task.entity_col].astype(int).to_numpy();cutoff=(tab[task.time_col].astype('int64')//10**9).to_numpy()
        keys=list(zip(entity.tolist(),cutoff.tolist()));assert len(set(keys))==len(keys)
        targets=[];ids=[];hops=[];edges=[];relations=[];token_times=[];padded=0
        for row,(e,c) in enumerate(keys):
            rr=raw[(raw.driverId==e)&(raw.date>pd.Timestamp(c,unit='s'))&(raw.date<=pd.Timestamp(c,unit='s')+pd.Timedelta(days=60))]
            targets.append(float(rr.positionOrder.mean()))
            root_id=int(offsets[names.index(task.entity_table)]+e)
            seed=int.from_bytes(hashlib.sha256(f'{root_id}:{c}:146'.encode()).digest()[:8],'little')
            chosen=temporal_context(root_id,c,adjacency,times,CONFIG['k'],seed)
            padded+=len(chosen)-len(set(chosen));ids.append(chosen)
            hops.append([0 if n==root_id else 1 if n in adjacency[root_id] else 2 for n in chosen])
            token_times.append([-1 if times[n] is None else times[n] for n in chosen])
            # Every duplicate slot gets its own incident edges. No last-slot overwrite.
            slots={}
            for j,n in enumerate(chosen):slots.setdefault(n,[]).append(j)
            es=[];rs=[]
            for j,n in enumerate(chosen):
                for other,rid in links[n]:
                    for dst in slots.get(other,[]):es.append((j,dst));rs.append(rid)
            edges.append(torch.tensor(es,dtype=torch.long).reshape(-1,2).t());relations.append(torch.tensor(rs,dtype=torch.long))
        np.testing.assert_allclose(targets,tab[task.target_col],rtol=1e-7,atol=1e-7);total_labels+=len(keys)
        ids=np.array(ids);tt=np.array(token_times);assert not ((tt!=-1)&(tt>cutoff[:,None])).any()
        ages=np.where(tt==-1,0,(cutoff[:,None]-tt)/86400).astype('float32');ages[:,0]=0
        packet=dict(entity=entity,cutoff=cutoff,target=np.array(targets),types=node_type[ids],indices=local_id[ids],
                    global_ids=ids,hops=np.array(hops),ages=ages,token_times=tt)
        np.savez_compressed(root/f'{split}.npz',**packet)
        torch.save(dict(edges=edges,relations=relations),root/f'{split}-edges.pt')
        summaries[split]=dict(queries=len(keys),future_tokens=0,padding_occurrences=padded,
                              cache_sha256=sha(root/f'{split}.npz'),edges_sha256=sha(root/f'{split}-edges.pt'))
    report=dict(status='PASS',config=CONFIG,graph=meta,graph_provenance='REUSED L143 graph and feature statistics, hash verified',
                independently_rebuilt_labels=total_labels,splits=summaries,node_types=names,edge_types=data.edge_types,
                temporal_contract='event time only; feature availability history not established')
    (root/'audit.json').write_text(json.dumps(report,indent=2));return report

class Contexts:
    def __init__(self,root,data):
        self.root=Path(root);self.data=data;self.names=data.node_types;self.audit=json.loads((self.root/'audit.json').read_text())
        self.packets={};self.edges={}
        for split in ['train','val','test']:
            for suffix,key in [('.npz','cache_sha256'),('-edges.pt','edges_sha256')]:
                assert sha(self.root/(split+suffix))==self.audit['splits'][split][key]
            self.packets[split]=dict(np.load(self.root/f'{split}.npz'))
            self.edges[split]=torch.load(self.root/f'{split}-edges.pt',weights_only=False)
    def batch(self,split,order,device):
        p=self.packets[split];bs=len(order);k=CONFIG['k'];types=torch.tensor(p['types'][order]);ids=torch.tensor(p['indices'][order])
        b={key:torch.tensor(p[src][order],device=device) for key,src in [('neighbor_types','types'),('neighbor_hops','hops'),('neighbor_times','ages')]}
        b['node_indices']=torch.tensor(p['global_ids'][order,0],device=device)
        b['labels']=torch.tensor(p['target'][order],dtype=torch.float32,device=device)
        b['grouped_tfs']={};b['grouped_indices']={}
        for t,name in enumerate(self.names):
            mask=types==t
            if mask.any():
                b['grouped_tfs'][t]=self.data[name].tf[ids[mask]].to(device)
                b['grouped_indices'][t]=mask.reshape(-1).nonzero().view(-1).tolist()
        b['flat_batch_idx']=torch.arange(bs).repeat_interleave(k).tolist();b['flat_nbr_idx']=list(range(k))*bs
        b['edge_index']=torch.cat([self.edges[split]['edges'][int(row)]+j*k for j,row in enumerate(order)],1).to(device)
        b['relation']=torch.cat([self.edges[split]['relations'][int(row)] for row in order]).to(device)
        b['batch']=torch.arange(bs,device=device).repeat_interleave(k)
        return b

def make_model(arm,data,stats):
    common=dict(col_names_dict={t:data[t].tf.col_names_dict for t in data.node_types})
    type_map={t:i for i,t in enumerate(data.node_types)};c=CONFIG
    if arm=='gnn':return CourseGNN(**common,stats=stats,type_map=type_map,relations=len(data.edge_types),width=c['width'],layers=c['layers'],dropout=c['dropout'])
    if arm!='relgt':raise ValueError(arm)
    return RelGT(**common,col_stats_dict=stats,node_type_map=type_map,num_nodes=data.num_nodes,max_neighbor_hop=3,
                 channels=c['width'],out_channels=1,local_num_layers=c['layers'],heads=4,
                 ff_dropout=c['dropout'],attn_dropout=c['dropout'],conv_type='full',num_centroids=c['centroids'],
                 global_dim=c['global_dim'],sample_node_len=c['k'])

def forward(model,arm,b):
    if arm=='gnn':return model(b)
    return model(*[b[k] for k in ['neighbor_types','node_indices','neighbor_hops','neighbor_times']],b,
                 edge_index=b['edge_index'],batch=b['batch']).view(-1)

def fit(root,prior,output,arm,seed,pilot=False):
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    _,task,data,stats,meta=task_data(prior);contexts=Contexts(root,data);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=make_model(arm,data,stats).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    clamp=np.percentile(contexts.packets['train']['target'],[2,98]);history=[];best=None;first_grad=None
    def evaluate(split,limit=None):
        model.eval();pred=[];rows=[]
        with torch.no_grad():
            for j in range(0,len(contexts.packets[split]['target']),CONFIG['batch']):
                if limit is not None and j>=limit*CONFIG['batch']:break
                order=np.arange(j,min(j+CONFIG['batch'],len(contexts.packets[split]['target'])))
                b=contexts.batch(split,order,device);y=forward(model,arm,b).clamp(*clamp)
                pred.extend(y.cpu().tolist());rows.extend(order.tolist())
        p=contexts.packets[split];return float(np.mean(np.abs(np.array(pred)-p['target'][rows]))),dict(entity=p['entity'][rows],cutoff=p['cutoff'][rows],target=p['target'][rows],pred=np.array(pred))
    for epoch in range(1,(1 if pilot else CONFIG['epochs'])+1):
        ep=time.perf_counter();model.train();order=np.random.default_rng(seed*1000+epoch).permutation(len(contexts.packets['train']['target']));count=0
        for j in range(0,len(order),CONFIG['batch']):
            if pilot and j>=3*CONFIG['batch']:break
            rows=order[j:j+CONFIG['batch']];b=contexts.batch('train',rows,device)
            optimizer.zero_grad();pred=forward(model,arm,b);loss=torch.nn.functional.l1_loss(pred,b['labels']);loss.backward()
            if first_grad is None:first_grad={n:int((~torch.isfinite(p.grad)).sum()) for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite loss/gradient: stop course comparison')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();count+=len(rows)
        train_seconds=time.perf_counter()-ep;ev=time.perf_counter();val,packet=evaluate('val',1 if pilot else None)
        history.append(dict(epoch=epoch,queries=count,val_mae=val,train_seconds=train_seconds,val_seconds=time.perf_counter()-ev))
        if select_config(list(range(len(history))),[h['val_mae'] for h in history])==len(history)-1:best=copy.deepcopy(model.state_dict())
        (out/'history.json').write_text(json.dumps(history));print(arm,seed,epoch,val,flush=True)
    model.load_state_dict(best);torch.save(best,out/'selected.pt');scores={};pack={}
    for split in (['val'] if pilot else ['val','test']):
        scores[split],p=evaluate(split,1 if pilot else None);pack.update({split+'_'+k:v for k,v in p.items()})
    # A second fixed-layout eval must agree within 1e-5 absolute (CUDA scatter ordering).
    score2,p2=evaluate('val',1 if pilot else None);np.testing.assert_allclose(p2['pred'],pack['val_pred'],atol=1e-5,rtol=0)
    np.savez_compressed(out/'predictions.npz',**pack)
    result=dict(run_id=str(uuid.uuid4()),arm=arm,seed=seed,config=CONFIG,pilot=pilot,history=history,scores=scores,
                parameters=sum(p.numel() for p in model.parameters()),selected_epoch=select_config(list(range(1,len(history)+1)),[h['val_mae'] for h in history]),
                first_batch_nonfinite_gradients=first_grad,repeat_eval_max_error=float(np.max(np.abs(p2['pred']-pack['val_pred']))),seconds=time.perf_counter()-start,
                checkpoint_sha256=sha(out/'selected.pt'),context_sha256={s:contexts.audit['splits'][s]['cache_sha256'] for s in contexts.packets},
                evidence='COURSE_EXPERIMENT' if not pilot else 'TIMING_PILOT',test='NOT_RUN' if pilot else 'SCORED')
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

def recover_completed_fits(root,prior):
    """Finalize five completed checkpoints after an overstrict exact-equality check.

    No optimizer or training loop is entered. Histories/checkpoints are retained.
    One numerical check changes to1e-5 absolute; no score-based decisions change.
    """
    torch.set_num_threads(1);root=Path(root);_,task,data,stats,meta=task_data(prior)
    contexts=Contexts(root/'prepared',data);device='cuda';clamp=np.percentile(contexts.packets['train']['target'],[2,98]);reports=[]
    for arm,seed in [('gnn',0),('relgt',0),('gnn',1),('relgt',1),('gnn',2)]:
        start=time.perf_counter();phase=f'fit-{arm}-{seed}';out=root/phase;assert not (out/'result.json').exists()
        history=json.loads((out/'history.json').read_text());assert len(history)==10 and all(h['queries']==7453 for h in history)
        checkpoint_hash=sha(out/'selected.pt');model=make_model(arm,data,stats).to(device)
        model.load_state_dict(torch.load(out/'selected.pt',map_location=device,weights_only=True));model.eval()
        state={k:v.clone() for k,v in model.state_dict().items()};pack={};scores={}
        def evaluate(split):
            preds=[]
            with torch.no_grad():
                for j in range(0,len(contexts.packets[split]['target']),128):
                    order=np.arange(j,min(j+128,len(contexts.packets[split]['target'])))
                    preds.extend(forward(model,arm,contexts.batch(split,order,device)).clamp(*clamp).cpu().tolist())
            return np.array(preds)
        for split in ['val','test']:
            p=contexts.packets[split];pred=evaluate(split);scores[split]=float(np.mean(np.abs(pred-p['target'])))
            pack.update({split+'_'+k:p[k] for k in ['entity','cutoff','target']});pack[split+'_pred']=pred
        repeated=evaluate('val');maxerror=float(np.max(np.abs(repeated-pack['val_pred'])))
        np.testing.assert_allclose(repeated,pack['val_pred'],rtol=0,atol=1e-5)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());assert sha(out/'selected.pt')==checkpoint_hash
        np.savez_compressed(out/'predictions.npz',**pack)
        result=dict(run_id=phase,arm=arm,seed=seed,config=CONFIG,pilot=False,history=history,scores=scores,
                    parameters=sum(p.numel() for p in model.parameters()),selected_epoch=select_config(list(range(1,11)),[h['val_mae'] for h in history]),
                    first_batch_nonfinite_gradients={},gradient_evidence='Original trainer checked every batch; all ten epochs completed',repeat_eval_max_error=maxerror,
                    seconds=json.loads((root/(phase+'-cost.json')).read_text())['seconds'],recovery_evaluation_seconds=time.perf_counter()-start,
                    checkpoint_sha256=checkpoint_hash,context_sha256={s:contexts.audit['splits'][s]['cache_sha256'] for s in contexts.packets},
                    evidence='COURSE_EXPERIMENT',test='SCORED',recovery='Evaluation only after exact CUDA-repeat assertion; no retraining or checkpoint change')
        (out/'result.json').write_text(json.dumps(result,indent=2));reports.append(result);print(phase,scores,maxerror,flush=True)
    return reports
