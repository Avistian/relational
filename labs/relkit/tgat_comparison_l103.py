"""Bounded matched-protocol course experiment, not a historical paper replay.
Model definitions are canonical and inlined by the notebook builder.
"""
# %% Shared event budget and fixed negative schedule
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
# The notebook builder removes these imports after inlining both model definitions.
from relkit.tgat_l103 import TGAT,NeighborFinder
from relkit.tgn_l102 import TGN,temporal_neighbors

def common_data(data,train_count=2000,eval_count=400):
    out={name:{k:v[:train_count if name=='train' else eval_count] for k,v in data[name].items()} for name in ['train','val','test']}
    full={k:np.concatenate([out[name][k] for name in ['train','val','test']]) for k in out['train']}
    out['full']=full
    return out

def fixed_negatives(data,seed):
    result={}
    for i,name in enumerate(['train','val','test']):
        pool=np.unique(data['train' if name=='train' else 'full']['v'])
        result[name]=np.random.RandomState(seed+1000*(i+1)).choice(pool,len(data[name]['u']))
    return result

# %% Shared trainer: three epochs, validation-only selection, pooled held-out AP

def run_comparison(nodes,edges,data,seed=0,epochs=3,output=None,device='cpu'):
    negatives=fixed_negatives(data,seed);records=[]
    for arm in ['TGAT','TGN']:
        torch.manual_seed(seed);np.random.seed(seed);begin=time.perf_counter()
        if arm=='TGAT':
            train_finder=NeighborFinder(data['train'],len(nodes),release=False,uniform=False)
            full_finder=NeighborFinder(data['full'],len(nodes),release=False,uniform=False)
            model=TGAT(train_finder,nodes,edges,layers=1).to(device)
        else:
            def make(events):return temporal_neighbors(events['u'],events['v'],events['t'],events['e'],len(nodes))
            train_finder,full_finder=make(data['train']),make(data['full'])
            model=TGN(nodes,edges).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        def run(name,training):
            events=data[name];ps=[];ns=[];losses=[];model.train(training)
            for start in range(0,len(events['u']),200):
                b={k:x[start:start+200] for k,x in events.items()};negative=negatives[name][start:start+200]
                with torch.set_grad_enabled(training):
                    if training:optimizer.zero_grad()
                    if arm=='TGAT':p,n=model.contrast(b['u'],b['v'],negative,b['t'],10)
                    else:p,n=model.probabilities(b['u'],b['v'],negative,b['t'],b['e'])
                    loss=nn.functional.binary_cross_entropy(p,torch.ones_like(p))+nn.functional.binary_cross_entropy(n,torch.zeros_like(n))
                    if training:loss.backward();optimizer.step()
                    if arm=='TGN':model.detach_state()
                ps.extend(p.detach().cpu().tolist());ns.extend(n.detach().cpu().tolist());losses.append(float(loss.detach()))
            y=np.r_[np.ones(len(ps)),np.zeros(len(ns))];scores=np.r_[ps,ns]
            return {'ap':float(average_precision_score(y,scores)),'auc':float(roc_auc_score(y,scores)),'loss':float(np.mean(losses))},np.asarray(ps),np.asarray(ns)
        best=-1;trace=[];selected=0
        for epoch in range(epochs):
            if arm=='TGAT':model.ngh_finder=train_finder
            else:model.finder=train_finder;model.reset_state()
            train,_,_=run('train',True)
            if arm=='TGAT':model.ngh_finder=full_finder
            else:model.finder=full_finder
            with torch.no_grad():val,_,_=run('val',False)
            trace.append({'epoch':epoch,'train':train,'val':val})
            if val['ap']>best:
                best=val['ap'];selected=epoch
                weights={k:v.detach().clone() for k,v in model.state_dict().items() if k not in ['node_features','edge_features','n_feat_th','e_feat_th','node_raw_embed.weight','edge_raw_embed.weight']}
                state=copy.deepcopy(model.snapshot()) if arm=='TGN' else None
        model.load_state_dict(weights,strict=False)
        if arm=='TGN':model.restore(state)
        with torch.no_grad():test,p,n=run('test',False)
        row={'arm':arm,'seed':seed,'selected_epoch':selected,'test':test,'trace':trace,'seconds':time.perf_counter()-begin,'trainable_used_note':'TGAT retains unused release parameters; parameter counts are not a matched budget'}
        if output:
            out=Path(output);out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{arm}-{seed}-predictions.npz',positive=p,negative_score=n,negative=negatives['test'],edges=data['test']['e'])
            torch.save({'weights':weights,'state':state},out/f'{arm}-{seed}.pt')
            (out/f'{arm}-{seed}.json').write_text(json.dumps(row,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True);records.append(row)
    return records
