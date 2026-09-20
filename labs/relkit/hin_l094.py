"""L094: executable survey taxonomy and data/protocol audit. No model training."""
from pathlib import Path
import hashlib
import numpy as np


def compose_path(relations, path):
    """Multiply source-row/target-column adjacencies; retain path multiplicity."""
    if not path:raise ValueError('A meta-path needs at least one relation')
    out=None
    for i,key in enumerate(path):
        if i and path[i-1][2]!=key[0]:raise ValueError('Adjacent endpoint types disagree')
        matrix=np.asarray(relations[key],dtype=np.int64)
        if matrix.ndim!=2 or np.any(matrix<0):raise ValueError('Expected nonnegative adjacency matrix')
        out=matrix.copy() if out is None else out@matrix
    return out


def classify_method(record):
    """Two independent axes, not a mutually exclusive family tree."""
    enc={'lookup':'lookup embedding','message_passing':'GNN','fixed':'fixed structural score'}
    routes={'explicit_path':'explicit meta-path','one_hop':'one-hop composition','branching':'meta-graph'}
    return [enc[record['encoder']],routes[record['route']]]


def compare_protocols(left,right):
    """Fail closed: unknown fields never prove compatibility, even when equal."""
    keys=['dataset_hash','task','split_hash','features','target_mask','selection','budget','metric','aggregation']
    unknown=[k for k in keys if left.get(k) in (None,'','UNKNOWN') or right.get(k) in (None,'','UNKNOWN')]
    differences=[k for k in keys if k not in unknown and left[k]!=right[k]]
    status='INCOMPARABLE' if differences else ('NOT_ESTABLISHED' if unknown else 'COMPARABLE')
    result={'status':status,'differences':differences}
    if unknown:result['unknown']=unknown
    return result


def table_arithmetic(row):
    """Check the printed partition under a declared single-direction convention."""
    ns=sum(row[k] for k in ['paper','author','field','venue','institute'])
    es=sum(row[k] for k in ['PA','PF','PV','AI','PP'])
    return {'node_sum':ns,'node_delta':ns-row['nodes'],'edge_sum':es,'edge_delta':es-row['edges']}


def count_graph(graph):
    """Release maps use target -> source -> relation -> target ID -> source IDs.
    Count distinct stored adjacency entries, not timestamp multiplicities.
    Do not silently discard FF edges, reverse edges, or unmatched reverse content.
    """
    nodes={k:len(v) for k,v in graph.node_feature.items()}
    rows=[];forward=0;stored=0;reverse_ok=True
    columns={k:0 for k in ['PA','PF','PV','AI','PP']}
    pairs={frozenset(['paper','author']):'PA',frozenset(['paper','field']):'PF',frozenset(['paper','venue']):'PV',frozenset(['author','affiliation']):'AI',frozenset(['paper']):'PP'}
    for target,sources in graph.edge_list.items():
        for source,rels in sources.items():
            for relation,adj in rels.items():
                n=sum(len(neighbors) for neighbors in adj.values());stored+=n
                reverse=relation.startswith('rev_')
                rows.append({'source':source,'target':target,'relation':relation,'edges':n,'reverse':reverse})
                if reverse:continue
                forward+=n
                column=pairs.get(frozenset([source,target]))
                if column:columns[column]+=n
                back=graph.edge_list.get(source,{}).get(target,{}).get('rev_'+relation,{})
                if sum(map(len,back.values()))!=n:reverse_ok=False
                for dst,srcs in adj.items():
                    for src,stamp in srcs.items():
                        if dst not in back.get(src,{}) or back[src][dst]!=stamp:reverse_ok=False
    columns.update({'nodes':sum(nodes.values()),'edges':stored,'paper':nodes.get('paper',0),'author':nodes.get('author',0),'field':nodes.get('field',0),'venue':nodes.get('venue',0),'institute':nodes.get('affiliation',0)})
    return {'nodes_by_type':nodes,'relations':rows,'forward_edges':forward,'all_stored_edges':stored,'listed_forward_edges':sum(columns[k] for k in ['PA','PF','PV','AI','PP']),'unlisted_forward_edges':forward-sum(columns[k] for k in ['PA','PF','PV','AI','PP']),'reverse_content_matches':reverse_ok,'table_columns':columns}


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
