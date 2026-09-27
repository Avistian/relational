"""Behavioral checks for the three learner-owned REG operations."""
import json
from pathlib import Path
import numpy as np

def check_edges(fn):
    got=fn([90,10,40],[40,None,90,40])
    np.testing.assert_array_equal(got,np.array([[0,2,3],[2,0,2]]))
    try:fn([90,90],[90])
    except ValueError:pass
    else:raise AssertionError('Duplicate primary key accepted')
    try:fn([90],[12])
    except ValueError:pass
    else:raise AssertionError('Unknown non-null foreign key accepted')
    assert fn([90],[None]).shape==(2,0)

def check_cutoff(fn):
    # The middle timeless dimension must not reset the seed cutoff.
    edges=[(0,1),(1,0),(1,2),(2,1),(1,3),(3,1)]
    times=[None,None,5,11]
    assert set(fn(edges,times,0,7,2))=={0,1,2}
    assert set(fn(edges,times,0,11,2))=={0,1,2,3}
    assert set(fn(edges,times,0,4,2))=={0,1}
    assert set(fn(edges,times,0,7,0))=={0}

def check_targets(fn):
    # Query rows 0 and 2 refer to the same entity at different times.
    target=np.array([3.,7.,11.]); input_id=np.array([2,0,1])
    np.testing.assert_array_equal(fn(target,input_id),[11.,3.,7.])
    np.testing.assert_array_equal(fn(target,np.array([],dtype=int)),[])

def red_edges(pk,fk):return np.array([[i,int(x)] for i,x in enumerate(fk) if x is not None]).T

def red_cutoff(edges,times,seed,cutoff,hops):return range(len(times))
def red_targets(target,input_id):return target

if __name__=='__main__':
    import sys
    if '--red' in sys.argv:
        failed=[]
        for check,fn in [(check_edges,red_edges),(check_cutoff,red_cutoff),(check_targets,red_targets)]:
            try:check(fn)
            except AssertionError:failed.append(check.__name__)
        assert len(failed)==3
        result={'status':'PASS','rejected_faults':failed}
    else:
        from relkit.rdl_l117 import foreign_key_edges,temporal_nodes,query_targets
        for check,fn in [(check_edges,foreign_key_edges),(check_cutoff,temporal_nodes),(check_targets,query_targets)]:check(fn)
        result={'status':'PASS','checks':['arbitrary key remapping/null/duplicate/dangling','root cutoff across two hops and equality boundary','query-row label identity']}
    Path(__file__).with_name('_'+('red' if '--red' in sys.argv else 'check')+'_l117_results.json').write_text(json.dumps(result,indent=2));print(result)
