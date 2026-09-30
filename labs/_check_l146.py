"""Behavioral contracts: invalid contexts, test-selected winners and shuffled pairs."""
import numpy as np

def check_selection(fn):
    configs=['a','b','c']
    assert fn(configs,[3.,2.,2.])=='b', 'First validation tie must win.'
    for vals in [[1,float('nan'),2],[],[1,2]]:
        try:fn(configs,vals)
        except ValueError:pass
        else:raise AssertionError('Invalid search accepted')

def check_pairs(fn):
    keys=[(1,10),(1,20),(2,10)];target=[1.,3.,5.]
    result=fn(keys,target,keys,[2.,5.,4.],keys[::-1],[5.,4.,3.])
    np.testing.assert_allclose(result,[-1.,1.,1.])
    for bad in [[(1,10),(1,10),(2,10)],[(1,10),(1,20),(8,10)]]:
        try:fn(keys,target,bad,[2,5,4],keys,[3,4,5])
        except ValueError:pass
        else:raise AssertionError('Misaligned/duplicate keys accepted')

def check_context(fn):
    # Future bridge must not expose an otherwise legal second-hop row.
    adj={0:[1,2],1:[0,3],2:[0,4],3:[1],4:[2],5:[]}
    times={0:None,1:9,2:4,3:2,4:3,5:None}
    assert fn(0,5,adj,times,1,146)==[0], 'A one-token context is only the root.'
    early=fn(0,5,adj,times,4,146)
    assert early[0]==0 and set(early)=={0,2,4} and len(early)==4
    late=fn(0,10,adj,times,4,146)
    assert 1 in late and len(late)==4
    assert fn(5,5,adj,times,4,146)==[5]*4, 'Isolated root cannot sample arbitrary rows.'
    assert fn(0,5,adj,times,4,146)==early, 'Sampling must be stable.'

def run():
    from relkit.comparison_l146 import select_config,paired_errors,temporal_context
    for test,fn in [(check_selection,select_config),(check_pairs,paired_errors),(check_context,temporal_context)]:test(fn)
    print('PASS: validation selection, keyed paired losses, owner-safe reachability')
if __name__=='__main__':run()
