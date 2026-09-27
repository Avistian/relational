"""Behavioral specification used unchanged by notebook checks."""
import json
from pathlib import Path
from relkit.temporal_l123 import eligible,sample_temporal,audit_sample,fixture,course_run

def check_eligible(fn):
    assert fn((8,8),8), 'Equality at cutoff is included by this contract'
    assert not fn((5,11),8), 'An old event may arrive after the query'
    assert not fn((12,12),8), 'Future events are invisible'
    assert fn(None,8), 'None here is an explicit timeless assumption'
    for stamp in [(None,8),(float('nan'),8),(9,7)]:
        try: fn(stamp,8)
        except ValueError: pass
        else: raise AssertionError('Reject missing/nonfinite or inconsistent timestamps')

def check_sample(fn):
    nodes,edges=fixture()
    a=fn(nodes,edges,('person',0),8,2)
    assert a['nodes']=={('person',0),('transfer',0),('memo',1)}, 'Use root cutoff at every hop, not parent time'
    assert set(a['edges'])=={0,4}, 'Reject future second-hop memo and delayed transfer'
    b=fn(nodes,edges,('person',0),12,2)
    assert len(b['nodes'])==6 and set(b['edges'])==set(range(5)), 'Later query gets its own neighborhood'
    assert fn(nodes,edges,('person',0),8,0)['nodes']=={('person',0)}
    # An edge can arrive later than both endpoints.
    delayed=[dict(edges[0],stamp=(4,9))]
    assert fn(nodes,delayed,('person',0),8,2)['edges']==[]
    # Incoming message direction matters; a reverse store is a separate relation.
    reverse=[dict(src=('person',0),dst=('transfer',0),kind='rev',stamp=(4,4))]
    assert fn(nodes,reverse,('person',0),8,2)['edges']==[]
    try: fn(nodes,edges,('memo',0),8,2)
    except ValueError: pass
    else: raise AssertionError('Reject a future root')

def check_audit(fn):
    nodes,edges=fixture()
    good={'root':('person',0),'cutoff':8,'hops':2,'nodes':{('person',0),('transfer',0),('memo',1)},'edges':[0,4]}
    assert fn(nodes,edges,good)['status']=='PASS'
    bads=[dict(good,nodes=good['nodes']|{('memo',0)},edges=[0,3,4]),
          dict(good,nodes=good['nodes']|{('transfer',2)},edges=[0,2,4]),
          dict(good,edges=[0,1,4]),dict(good,edges=[99])]
    for bad in bads:
        try:fn(nodes,edges,bad)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Audit accepted future/late/missing endpoint/invalid identity')

if __name__=='__main__':
    check_eligible(eligible);check_sample(sample_temporal);check_audit(audit_sample)
    r=course_run();Path(__file__).with_name('_check_l123_results.json').write_text(json.dumps(r,indent=2));print(r)
