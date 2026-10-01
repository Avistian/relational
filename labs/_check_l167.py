"""Learner feedback: expected outcomes come from hand-worked examples."""
def check167(summary_fn, support_fn, auc_fn):
    import numpy as np
    queries=[[7,10],[7,20],[9,10]]
    events=[[7,2,4],[7,8,8],[7,10,100],[7,14,12],[9,12,99]]
    np.testing.assert_allclose(summary_fn(queries,events),[[2,6],[4,31],[0,0]])
    np.testing.assert_allclose(summary_fn(queries,events[::-1]),[[2,6],[4,31],[0,0]])
    np.testing.assert_array_equal(support_fn([[7,2],[8,5],[9,10]],[9,12,10],10),[0])
    keys=[[1,10],[1,20],[2,10],[2,20]]
    assert auc_fn(keys,[0,1,0,1],keys[::-1],[1.,.5,.5,.1])==.875
    for bad_keys,bad_p in [(keys[:-1],[.1,.5,.5]),(keys+[keys[0]],[.1,.5,.5,1.,.1]),(keys,[.1,.5,float('nan'),1.])]:
        try:auc_fn(keys,[0,1,0,1],bad_keys,bad_p)
        except ValueError:pass
        else:raise AssertionError('Invalid prediction population accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.transfer_l167 import temporal_summary,eligible_support,keyed_auc
    print(check167(temporal_summary,eligible_support,keyed_auc))
