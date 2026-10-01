"""Visible course audits; no historical BART/GCN implementation is implied."""
def reachable_rows(nodes, edges, root, hops, allowed_tables=None):
    """At most hops on an undirected, table-filtered graph; include the root."""
    valid_id = lambda x: isinstance(x,str) and bool(x) and x.strip()==x
    if not isinstance(nodes,dict) or not nodes or not all(valid_id(k) and valid_id(v) for k,v in nodes.items()):
        raise ValueError('Canonical nonempty node and table IDs required')
    if type(hops) is not int or hops<0 or root not in nodes:
        raise ValueError('Known root and nonnegative integer hops required')
    if not isinstance(edges,list):raise ValueError('Edges must be a list')
    seen_edges=set()
    for edge in edges:
        if not isinstance(edge,(list,tuple)) or len(edge)!=2 or not all(isinstance(x,str) and x in nodes for x in edge) or edge[0]==edge[1]:
            raise ValueError('Edges need two distinct known endpoints')
        pair=tuple(sorted(edge))
        if pair in seen_edges:raise ValueError('Duplicate undirected edge')
        seen_edges.add(pair)
    if allowed_tables is None:allowed=set(nodes.values())
    else:
        if not isinstance(allowed_tables,list) or not allowed_tables or not all(isinstance(t,str) and t in nodes.values() for t in allowed_tables) or len(set(allowed_tables))!=len(allowed_tables):
            raise ValueError('Unique known table list required')
        allowed=set(allowed_tables)
    if nodes[root] not in allowed:raise ValueError('Root table must be retained')
    adjacency={n:set() for n,t in nodes.items() if t in allowed}
    for a,b in seen_edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b);adjacency[b].add(a)
    visited={root};frontier={root}
    for _ in range(hops):
        frontier={v for u in frontier for v in adjacency[u]}-visited
        visited.update(frontier)
        if not frontier:break
    return sorted(visited)

def token_budget(lengths, limit):
    """Dense attention pair proxies, not runtime or total LM/GNN memory."""
    if not isinstance(lengths,list) or not lengths or any(type(n) is not int or n<0 for n in lengths):
        raise ValueError('Nonempty list of nonnegative integer token counts required')
    if type(limit) is not int or limit<=0:raise ValueError('Positive integer limit required')
    kept=[min(n,limit) for n in lengths]
    return dict(rows=len(lengths),total_tokens=sum(lengths),retained_tokens=sum(kept),
                dropped_tokens=sum(lengths)-sum(kept),overflow_rows=[i for i,n in enumerate(lengths) if n>limit],
                whole_table_pairs=sum(lengths)**2,row_pairs=sum(n*n for n in lengths),retained_row_pairs=sum(n*n for n in kept))

def evidence_verdict(record):
    """Consistency screen for declared evidence; requires human authentication."""
    keys=['multitable','heldout_database','pretraining_audited','adaptation_legal','matched_baseline','predictions_verified']
    codes=['MULTITABLE_EVALUATION','HELD_OUT_DATABASE','PRETRAINING_PROVENANCE','LEGAL_ADAPTATION','MATCHED_BASELINE','VERIFIED_PREDICTIONS']
    if not isinstance(record,dict) or set(record)!=set(keys) or any(type(record[k]) is not bool for k in keys):
        raise ValueError('Exactly six Boolean evidence declarations required')
    missing=[code for key,code in zip(keys,codes) if not record[key]]
    level='RECONSTRUCTION_ONLY' if not record['multitable'] else 'MULTITABLE_ONLY'
    if not missing:level='TRANSFER_REVIEW_ELIGIBLE'
    return dict(supported=level,transfer='NOT_ESTABLISHED',missing=missing)
