"""Independent tiny contracts: typed fanout, query identity, weighted timing and spend."""
import json, math
from pathlib import Path
import numpy as np

def check_bound(fn):
    schema=[('orders','by','users'),('items','in','orders'),('users','rev','orders')]
    r=fn(schema,'users',2,[3,2])
    assert r['frontiers']==[{'users':2},{'orders':6},{'items':12,'users':12}]
    assert r['node_occurrences']==32 and r['edge_occurrences']==30
    assert fn(schema,'users',2,[0,2])['node_occurrences']==2
    try:fn(schema,'users',2,[-1])
    except ValueError:pass
    else:raise AssertionError('All-neighbor sentinel cannot be a finite bound')

def check_audit(fn):
    times={'u':[1,1],'v':[4,8]};owners={'u':[0,1],'v':[0,1]}
    edges=[('v','u',[[0,1],[0,1]])]
    assert fn(times,owners,[5,10],edges)['nodes']==4
    for t,o,e in [({'u':[1,1],'v':[6,8]},owners,edges),(times,owners,[('v','u',[[1],[0]])]),(times,{'u':[0,2],'v':[0,1]},edges)]:
        try:fn(t,o,[5,10],e)
        except ValueError:pass
        else:raise AssertionError('Future node or mixed query accepted')
    assert fn({'u':[5]}, {'u':[0]}, [5], [])['nodes']==1

def check_summary(fn):
    rows=[dict(queries=2,sample_s=1.,transfer_s=0.,step_s=1.,audit_s=9.,peak_allocated_bytes=100,peak_reserved_bytes=200),dict(queries=6,sample_s=1.,transfer_s=0.,step_s=3.,audit_s=9.,peak_allocated_bytes=120,peak_reserved_bytes=256)]
    r=fn(rows)
    assert r['queries']==8 and math.isclose(r['queries_per_second'],8/6) and r['peak_allocated_bytes']==120
    assert math.isclose(r['audited_queries_per_second'],8/24)
    # Negative observations must not hide behind a positive aggregate.
    for field in ['sample_s', 'transfer_s', 'step_s', 'audit_s']:
        broken=[dict(rows[0], **{field:-1.}), dict(rows[1], **{field:2.})]
        try:fn(broken)
        except ValueError:pass
        else:raise AssertionError('Negative batch timing accepted: '+field)
    for count in [-1, 0, 1.5, True, float('nan'), float('inf')]:
        try:fn([dict(rows[0],queries=count)])
        except ValueError:pass
        else:raise AssertionError('Invalid query count accepted')
    try:fn([])
    except ValueError:pass
    else:raise AssertionError('Empty timing accepted')

def check_budget(fn):
    assert fn(2,3,4,10)==9
    for args in [(2,3,6,10),(-1,0,1,10),(0,0,float('nan'),10)]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid budget accepted')

if __name__=='__main__':
    import importlib.util
    path=Path(__file__).parent/'relkit/scale_l134.py'
    assert path.exists(),'Missing visible sampling implementation'
    from relkit.scale_l134 import frontier_bound,audit_queries,profile_summary,reserve_cost
    for check,fn in [(check_bound,frontier_bound),(check_audit,audit_queries),(check_summary,profile_summary),(check_budget,reserve_cost)]:check(fn)
    r=dict(status='PASS',contracts=4,negative_cases=18)
    Path(__file__).with_name('_check_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
