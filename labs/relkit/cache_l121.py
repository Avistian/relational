"""Cache deterministic feature conversion, never learned embeddings or dropout.
Inputs are equivalent to L118 collate_graphs, including node/edge/table order.
"""
import numpy as np
import torch
from relkit.cvitkovic_l118 import encode_features,computation_edges

def cache_record(record,info):
    identity,(edges,types,edge_types,features,label)=record
    return identity,encode_features(features,info),torch.tensor(types),computation_edges(len(types),edges),label

def collate_cached(records):
    all_features={};types=[];edges=[];batch=[];labels=[];ids=[];offset=0
    for graph,(identity,features,node_types,edge,label) in enumerate(records):
        for table,pair in features.items():all_features.setdefault(table,[]).append(pair)
        types.append(node_types);edges.append(edge+offset);batch.append(torch.full((len(node_types),),graph,dtype=torch.long))
        offset+=len(node_types);ids.append(identity);labels.append(label)
    features={t:tuple(torch.cat([v[i] for v in pairs]) for i in [0,1]) for t,pairs in all_features.items()}
    return (features,torch.cat(types),torch.cat(edges,dim=1),torch.cat(batch),len(records)),torch.tensor(labels),np.array(ids)
