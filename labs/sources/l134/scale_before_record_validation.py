"""Visible scale contracts; NumPy-only learner functions also run without PyG."""
import math
import numpy as np

def frontier_bound(schema, seed_type, batch_size, fanouts):
    """Incoming relation expansion, with no collisions; counts occurrences, not unique IDs."""
    if batch_size < 1 or int(batch_size)!=batch_size or any(f < 0 or int(f)!=f for f in fanouts):
        raise ValueError('Positive batch and finite nonnegative integer fanouts required')
    if len(set(schema))!=len(schema):
        raise ValueError('Duplicate relation')
    frontiers=[{seed_type:int(batch_size)}]
    for fanout in fanouts:
        following={}
        for source,relation,destination in schema:
            n=frontiers[-1].get(destination,0)*int(fanout)
            if n:following[source]=following.get(source,0)+n
        frontiers.append(following)
    nodes=sum(sum(f.values()) for f in frontiers)
    return dict(frontiers=frontiers,node_occurrences=nodes,edge_occurrences=nodes-int(batch_size))

def audit_queries(times, owners, cutoffs, edges):
    """Verify root ownership and inclusive source cutoff; missing static time is explicit None."""
    cutoffs=np.asarray(cutoffs);n=0
    for kind,group in owners.items():
        group=np.asarray(group,dtype=np.int64)
        if np.any(group<0) or np.any(group>=len(cutoffs)):
            raise ValueError('Invalid query owner')
        if times[kind] is not None:
            stamp=np.asarray(times[kind])
            if stamp.shape!=group.shape or np.any(stamp>cutoffs[group]):
                raise ValueError('Future node or mismatched timestamp shape')
        n+=len(group)
    count=0
    for source,destination,index in edges:
        a,b=np.asarray(index,dtype=np.int64)
        if np.any(a<0) or np.any(b<0) or np.any(a>=len(owners[source])) or np.any(b>=len(owners[destination])):
            raise ValueError('Edge coordinate outside sampled table')
        if np.any(np.asarray(owners[source])[a]!=np.asarray(owners[destination])[b]):
            raise ValueError('Cross-query message')
        count+=len(a)
    return dict(nodes=n,edges=count)

def profile_summary(records):
    """Ratio of totals, with audit overhead reported separately; callers exclude warmups."""
    if not records:raise ValueError('No measured batches')
    timing={k:sum(float(r[k]) for r in records) for k in ['sample_s','transfer_s','step_s','audit_s']}
    if any(not math.isfinite(v) or v<0 for v in timing.values()):raise ValueError('Invalid timing')
    seconds=sum(timing[k] for k in ['sample_s','transfer_s','step_s'])
    if seconds<=0:raise ValueError('Zero duration')
    queries=sum(int(r['queries']) for r in records)
    return dict(batches=len(records),queries=queries,**timing,queries_per_second=queries/seconds,
        audited_queries_per_second=queries/(seconds+timing['audit_s']),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records))

def reserve_cost(spent, reserved, planned, cap=10.):
    values=[spent,reserved,planned,cap]
    if any(not math.isfinite(v) or v<0 for v in values) or spent+reserved+planned>cap:
        raise ValueError('Aggregate cost reservation refused')
    return spent+reserved+planned


def inspect_batch(batch, graph, entity):
    """Bind the learner audit to actual sampler output plus original-edge/time identities."""
    from relkit.batch_audit_l123 import audit_batch
    checked=audit_batch(batch,graph,entity)
    portable=audit_queries({k:batch[k].time.numpy() if 'time' in batch[k] else None for k in batch.node_types},
        {k:batch[k].batch.numpy() for k in batch.node_types},batch[entity].seed_time.numpy(),
        [(a,c,batch[(a,b,c)].edge_index.numpy()) for a,b,c in batch.edge_types])
    assert portable['edges']==checked['edge_occurrences']
    return checked
