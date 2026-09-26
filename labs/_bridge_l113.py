"""Teaching bridge: same L112 GCN, full arxiv, random induced batches.
Random partitions deliberately expose cut-edge costs; they are not METIS or
OGB Table4 reproduction. Same initialization and label passes; optimizer steps differ.
"""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relkit.ogb_l112 import GCN,load_arxiv,normalized_adjacency,evaluate
from relkit.scaling_l113 import induced_edges

def bridge(x,edge,y,split,epochs=2,seed=113):
    torch.manual_seed(seed)
    base=GCN();full_adj=normalized_adjacency(edge,len(y))
    # Label-blind fixed grouping. All nodes belong to exactly one batch.
    groups=torch.randperm(len(y)).tensor_split(4)
    batches=[];mask=torch.zeros(len(y),dtype=torch.bool);mask[split['train']]=True
    for ids in groups:
        local=induced_edges(edge,ids,len(y))
        batches.append((ids,normalized_adjacency(local,len(ids)),mask[ids]))
    rows=[]
    for regime in ['full','mini']:
        torch.manual_seed(seed);model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        start=time.perf_counter();steps=0
        for epoch in range(epochs):
            model.train()
            current=[(torch.arange(len(y)),full_adj,mask)] if regime=='full' else batches
            for ids,a,m in current:
                optimizer.zero_grad();loss=F.nll_loss(model(x[ids],a)[m],y[ids][m]);loss.backward();optimizer.step();steps+=1
        scores,pred=evaluate(model,x,full_adj,y,split)
        rows.append({'regime':regime,'seed':seed,'epochs':epochs,'optimizer_steps':steps,'seconds':time.perf_counter()-start,**scores,
                     'max_batch_nodes':len(y) if regime=='full' else max(len(a) for a in groups),
                     'normalized_entries_per_label_pass':full_adj.values().numel() if regime=='full' else sum(a.values().numel() for _,a,_ in batches)})
    return {'status':'TEACHING_ONLY','scope':'Full arxiv, same GCN initialization, two label passes; random partitions, local degrees, different BN populations and optimizer steps; no sampler superiority claim','rows':rows}
if __name__=='__main__':
    torch.set_num_threads(1);x,e,y,s,a=load_arxiv(Path(__file__).parent/'data/l112');r=bridge(x,e,y,s)
    (Path(__file__).parent/'_bridge_l113_results.json').write_text(json.dumps(r,indent=2));print(r)
