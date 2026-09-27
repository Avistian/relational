"""Diagnostic learner checks: information, relation semantics, time and identity."""
import json
from pathlib import Path
import numpy as np

def check_ceiling(fn):
    assert fn([(2, 4), (2, 4)], [0, 1]) == .5, 'A collided pair cannot both be correct'
    assert fn([(0,), (0,), (0,), (1,), (1,)], [0, 0, 1, 1, 1]) == .8, 'Weight groups by rows'
    assert fn([(1,), (2,)], ['yes', 'no']) == 1., 'Distinct rows can separate observed labels'
    for x,y in [([], []), ([(1,)], [0,1])]:
        try: fn(x,y)
        except ValueError: pass
        else: raise AssertionError('Reject empty or misaligned observations')

def check_typed(fn):
    edges=[(0,2,'buy'),(1,2,'refund')]
    np.testing.assert_array_equal(fn([10.,4.,0.],edges,{'buy':1.,'refund':-1.}),[0,0,6])
    np.testing.assert_array_equal(fn([10.,4.,0.],[(0,2,'refund'),(1,2,'buy')],{'buy':1.,'refund':-1.}),[0,0,-6])
    np.testing.assert_array_equal(fn([2.,0.],[(0,1,'r'),(0,1,'r')],{'r':3.}),[0,12])
    # Permutation equivariance: rename nodes without changing the computation.
    np.testing.assert_array_equal(fn([0.,10.,4.],[(1,0,'buy'),(2,0,'refund')],{'buy':1.,'refund':-1.}),[6,0,0])

def check_time(fn):
    edges=[(0,1),(1,2),(1,3),(1,4)]
    event=[None,None,5,6,11];available=[None,None,9,6,11]
    assert fn(edges,event,available,0,7,2)==[0,1,3], 'Late-arriving and future rows must be excluded'
    assert fn(edges,event,available,0,9,2)==[0,1,2,3], 'Equality at availability cutoff is allowed'
    assert fn(edges,event,available,0,7,1)==[0,1], 'Respect hop budget'
    assert fn(edges,event,available,0,7,0)==[0], 'Seed only with zero hops'
    assert fn([(0,1)],[None,3],[None,4],0,4,1)==[0,1]

def check_mae(fn):
    ids=['driver90@7','driver10@7','driver90@12']
    assert fn([ids[2],ids[0],ids[1]],[10,4,5],ids,[3,7,11])==4/3, 'Join by full query ID, not position'
    for pi,p,ti,t in [(['a','a'],[1,2],['a','b'],[1,2]),(['a'],[1],['b'],[1]),(['a'],[np.nan],['a'],[1]),(['a'],[1,2],['a'],[1])]:
        try: fn(pi,p,ti,t)
        except ValueError: pass
        else: raise AssertionError('Reject ambiguous, missing or nonfinite predictions')

if __name__=='__main__':
    from relkit import synthesis_l119 as m
    checks=[(check_ceiling,m.collision_ceiling),(check_typed,m.typed_sum),(check_time,m.eligible_nodes),(check_mae,m.aligned_mae)]
    for check,fn in checks:check(fn)
    result=dict(status='PASS',checks=[c.__name__ for c,f in checks])
    (Path(__file__).parent/'_check_l119_results.json').write_text(json.dumps(result,indent=2))
    print(result)
