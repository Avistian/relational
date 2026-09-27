"""Visible root marking, query-owned labels and independent MAP for Lesson 132."""
import torch

def mark_roots(x, batch_size, marker):
    """Add ONE shared role vector to root occurrences, without mutating encoded inputs."""
    if x.ndim != 2 or marker.shape != (1,x.shape[1]) or not 0 <= batch_size <= len(x):
        raise ValueError('Expected N×d rows, 1×d marker and valid root prefix')
    return torch.cat((x[:batch_size]+marker,x[batch_size:]),dim=0)

def candidate_targets(owner, node_id, positive_owner, positive_id, batch_size):
    """A destination is positive only for its query owner; global ID alone is insufficient."""
    if batch_size <= 0 or owner.shape != node_id.shape or positive_owner.shape != positive_id.shape:
        raise ValueError('Aligned query/node arrays and positive batch size required')
    for a in [owner,positive_owner]:
        if a.ndim!=1 or a.dtype!=torch.long or (a<0).any() or (a>=batch_size).any():
            raise ValueError('Owner index outside current query batch')
    for a in [node_id,positive_id]:
        if a.ndim!=1 or a.dtype!=torch.long or (a<0).any():raise ValueError('Invalid node ID')
    return torch.isin(owner+batch_size*node_id,positive_owner+batch_size*positive_id).float()

def mean_average_precision(predictions, positives, k):
    """Independent MAP@k for nonempty query ground truth; one contribution per query."""
    import math
    if k<=0 or len(predictions)!=len(positives) or len(predictions)==0:
        raise ValueError('Nonempty aligned queries and positive k required')
    values=[]
    for ranked,truth in zip(predictions,positives):
        ranked=list(ranked);truth=set(truth)
        if len(ranked)!=k or len(set(ranked))!=k or not truth:
            raise ValueError('Exactly k unique predictions and nonempty truth required')
        hits=0;terms=[]
        for rank,node in enumerate(ranked,1):
            if node in truth:hits+=1;terms.append(hits/rank)
        values.append(math.fsum(terms)/min(k,len(truth)))
    return math.fsum(values)/len(values)

# %% A hand-checkable expressiveness witness
def cycle_witness():
    """Six-cycle versus two triangles, constant features; root-marked walk returns differ."""
    a=torch.zeros(6,6,dtype=torch.float64);b=torch.zeros_like(a)
    for matrix,cycles in [(a,[list(range(6))]),(b,[[0,1,2],[3,4,5]])]:
        for cycle in cycles:
            for i,u in enumerate(cycle):
                v=cycle[(i+1)%len(cycle)];matrix[u,v]=matrix[v,u]=1
    rows=[]
    for name,adj in [('six-cycle',a),('two-triangles',b)]:
        plain=torch.ones(6,1,dtype=torch.float64)
        marked=mark_roots(torch.zeros_like(plain),1,torch.ones(1,1,dtype=torch.float64))
        trace=[marked[:,0].tolist()]
        for _ in range(3):plain=adj@plain;marked=adj@marked;trace.append(marked[:,0].tolist())
        rows.append(dict(graph=name,plain=plain[:,0].tolist(),marked_steps=trace,root_return=float(marked[0,0])))
    return rows

# %% Explicit RelBench forward pass with the live marker function
def identity_forward(model,batch,source_type,destination_type,marker_enabled=True):
    """Same ordering as the released forward_dst_readout; only the marker is switchable."""
    if model.id_awareness_emb is None:raise ValueError('Model needs identity awareness')
    seed_time=batch[source_type].seed_time
    x=model.encoder(batch.tf_dict)
    if marker_enabled:x[source_type]=mark_roots(x[source_type],len(seed_time),model.id_awareness_emb.weight)
    time=model.temporal_encoder(seed_time,batch.time_dict,batch.batch_dict)
    for kind,value in time.items():x[kind]=x[kind]+value
    for kind,embedding in model.embedding_dict.items():x[kind]=x[kind]+embedding(batch[kind].n_id)
    x=model.gnn(x,batch.edge_index_dict)
    return model.head(x[destination_type])
