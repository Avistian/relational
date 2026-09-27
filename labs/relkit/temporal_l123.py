"""Inspectable query-time graph operations; explicit temporal contracts."""
# %% PROVIDED: dependencies
import json
import math

# %% PROVIDED: worked fixture

def fixture():
    # None means explicitly assumed timeless, not an unknown timestamp.
    nodes = {('person',0):None, ('transfer',0):(4,4),
             ('transfer',1):(9,9), ('transfer',2):(5,11),
             ('memo',0):(12,12), ('memo',1):(7,7)}
    edges = [dict(src=('transfer',i),dst=('person',0),kind='receiver',stamp=None) for i in range(3)]
    edges += [dict(src=('memo',i),dst=('transfer',0),kind='describes',stamp=None) for i in range(2)]
    return nodes,edges


def course_run():
    nodes,edges=fixture();reports=[]
    for cutoff in [8,10,12]:
        sample=sample_temporal(nodes,edges,('person',0),cutoff,2)
        report=audit_sample(nodes,edges,sample)
        reports.append(dict(cutoff=cutoff,nodes=len(sample['nodes']),edges=len(sample['edges']),audit=report))
    return dict(status='PASS',runs=reports,scope='Synthetic event/availability fixture; no predictive metric')

# %% Task 1: visibility at one query cutoff
def eligible(stamp, cutoff):
    if not math.isfinite(cutoff):
        raise ValueError('Cutoff must be finite')
    if stamp is None:
        return True
    event, available = stamp
    if any(x is None or not math.isfinite(x) for x in stamp) or available < event:
        raise ValueError('Explicit finite event <= availability times required')
    return event <= cutoff and available <= cutoff

# %% Task 2: incoming multi-hop temporal neighborhood
def sample_temporal(nodes, edges, root, cutoff, hops=2):
    if not isinstance(hops,int) or hops < 0:
        raise ValueError('Nonnegative integer hop count required')
    if root not in nodes or not eligible(nodes[root],cutoff):
        raise ValueError('Root does not exist at query time')
    incoming = {}
    for i,e in enumerate(edges):
        if e['src'] not in nodes or e['dst'] not in nodes:
            raise ValueError('Unknown edge endpoint')
        incoming.setdefault(e['dst'],[]).append(i)
    seen, frontier, chosen = {root}, {root}, set()
    for _ in range(hops):
        following = set()
        for target in sorted(frontier):
            for i in incoming.get(target,[]):
                e=edges[i];source=e['src']
                if (eligible(nodes[source],cutoff) and eligible(nodes[target],cutoff)
                        and eligible(e['stamp'],cutoff)):
                    chosen.add(i)
                    if source not in seen: following.add(source)
        seen.update(following);frontier=following
    return dict(root=root,cutoff=cutoff,hops=hops,nodes=seen,edges=sorted(chosen))

# %% Task 3: validate a returned temporal neighborhood
def audit_sample(nodes, edges, sample):
    cutoff=sample['cutoff']; selected=sample['nodes']
    if sample['root'] not in selected:
        raise ValueError('Root missing')
    for node in selected:
        if node not in nodes or not eligible(nodes[node],cutoff):
            raise ValueError('Unknown or future/late node')
    for index in sample['edges']:
        if index < 0 or index >= len(edges):
            raise ValueError('Unknown edge identity')
        edge=edges[index]
        if edge['src'] not in selected or edge['dst'] not in selected:
            raise ValueError('Edge crosses neighborhood boundary')
        if not eligible(edge['stamp'],cutoff):
            raise ValueError('Future/late edge')
    # This audit proves visibility and endpoint closure, not completeness/reachability.
    return dict(status='PASS',nodes=len(selected),edges=len(sample['edges']))

