"""Visible course contracts, not a reconstruction of proprietary KumoRFM weights.

Integer times share one arbitrary unit. Facts are visible at <= cutoff; context
anchors must be strictly earlier. These explicit conventions are course choices.
"""
def eligible_context(context, query):
    """Select mature, arrived labels without admitting the query's target."""
    def valid_time(x):
        return type(x) is int
    def valid_entity(x):
        return isinstance(x,str) and bool(x) and x.strip()==x
    if (not isinstance(query,dict) or set(query)!={'entity','cutoff'} or
        not valid_entity(query['entity']) or not valid_time(query['cutoff'])):
        raise ValueError('query requires canonical entity and integer cutoff')
    if not isinstance(context,list):
        raise ValueError('context must be a list')
    seen=set(); selected=[]
    for row in context:
        if (not isinstance(row,dict) or set(row)!={'entity','cutoff','label_end','available_at','label'} or
            not valid_entity(row['entity']) or
            not all(valid_time(row[k]) for k in ['cutoff','label_end','available_at']) or
            type(row['label']) is not int or row['label'] not in [0,1] or
            row['label_end']<row['cutoff'] or row['available_at']<row['label_end']):
            raise ValueError('invalid context record or inconsistent label clock')
        key=(row['entity'],row['cutoff'])
        if key in seen:
            raise ValueError('duplicate context key')
        seen.add(key)
        if (row['cutoff']<query['cutoff'] and row['label_end']<=query['cutoff'] and
            row['available_at']<=query['cutoff']):
            selected.append(dict(row))
    return sorted(selected,key=lambda r:(r['cutoff'],r['entity']))

def visible_graph(rows, edges, root, cutoff, hops):
    """At-most-hop reachability after filtering facts by their owner's cutoff.

Rows intentionally contain no label/value column. Edges are static course FK
links; their availability is inherited from both endpoint rows. This does not
model changing keys, corrected values or production database history.
"""
    if type(cutoff) is not int or type(hops) is not int or hops<0:
        raise ValueError('integer cutoff and nonnegative integer hops required')
    if not isinstance(rows,list) or not isinstance(edges,list):
        raise ValueError('rows and edges must be lists')
    by_id={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!={'id','event_at','available_at'} or
            not isinstance(row['id'],str) or not row['id'] or row['id'].strip()!=row['id'] or
            type(row['available_at']) is not int or
            (row['event_at'] is not None and (type(row['event_at']) is not int or row['available_at']<row['event_at']))):
            raise ValueError('invalid row, extra feature or inconsistent fact clock')
        if row['id'] in by_id:
            raise ValueError('duplicate row id')
        by_id[row['id']]=row
    if not isinstance(root,str) or root not in by_id:
        raise ValueError('unknown root')
    legal={k for k,r in by_id.items() if r['available_at']<=cutoff and (r['event_at'] is None or r['event_at']<=cutoff)}
    if root not in legal:
        raise ValueError('root is unavailable at its cutoff')
    adjacency={k:set() for k in legal};seen_edges=set()
    for edge in edges:
        if (not isinstance(edge,list) or len(edge)!=2 or any(not isinstance(x,str) or x not in by_id for x in edge) or edge[0]==edge[1]):
            raise ValueError('invalid edge')
        a,b=edge;pair=tuple(sorted(edge))
        if pair in seen_edges:
            raise ValueError('duplicate undirected edge')
        seen_edges.add(pair)
        if a in legal and b in legal:
            adjacency[a].add(b);adjacency[b].add(a)
    reached={root};frontier={root}
    for _ in range(hops):
        frontier={other for node in frontier for other in adjacency[node]}-reached
        reached.update(frontier)
        if not frontier:
            break
    return sorted(reached)

def keyed_auc(truth, predictions):
    """Rank-based binary AUROC with exact key alignment and averaged tie ranks."""
    import math
    def indexed(records,field):
        if not isinstance(records,list) or not records:
            raise ValueError('nonempty list required')
        out={}
        for r in records:
            if (not isinstance(r,dict) or set(r)!={'entity','cutoff',field} or
                not isinstance(r['entity'],str) or not r['entity'] or r['entity'].strip()!=r['entity'] or type(r['cutoff']) is not int):
                raise ValueError('invalid keyed record')
            key=(r['entity'],r['cutoff']);v=r[field]
            if key in out:
                raise ValueError('duplicate full query key')
            if field=='label':
                if type(v) is not int or v not in [0,1]:
                    raise ValueError('binary integer labels required')
            elif type(v) not in [int,float] or not math.isfinite(v):
                raise ValueError('finite numeric scores required')
            out[key]=v
        return out
    labels=indexed(truth,'label');scores=indexed(predictions,'score')
    if set(labels)!=set(scores):
        raise ValueError('prediction and truth key sets differ')
    n_positive=sum(labels.values());n_negative=len(labels)-n_positive
    if not n_positive or not n_negative:
        raise ValueError('AUROC requires both classes')
    ordered=sorted(labels,key=lambda key:scores[key]);rank_sum=0.;start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered) and scores[ordered[end]]==scores[ordered[start]]:
            end+=1
        average_rank=(start+1+end)/2
        rank_sum+=average_rank*sum(labels[k] for k in ordered[start:end])
        start=end
    return (rank_sum-n_positive*(n_positive+1)/2)/(n_positive*n_negative)
