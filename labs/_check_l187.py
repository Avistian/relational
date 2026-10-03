"""Behavioral checks, including wrong privacy units and unstable global clipping."""
import numpy as np
import pandas as pd


def checks(owned_counts, bounded_histogram, release_scale):
    db={'drivers':pd.DataFrame({'driverId':[7,9,11]}),
        'results':pd.DataFrame({'driverId':[7,7,9]}),
        'qualifying':pd.DataFrame({'driverId':[7]}),
        'standings':pd.DataFrame({'driverId':[9,9]})}
    out=owned_counts(db).set_index('driverId')
    assert list(out.loc[[7,9,11],'owned_rows'])==[4,4,1], 'Count each owned record, including driver'
    e=pd.DataFrame({'resultId':[4,1,3,2,5], 'driverId':[7,7,9,7,9], 'constructorId':[20,10,20,20,10]})
    h=bounded_histogram(e,[10,20,30],2)
    assert np.array_equal(h,[2,2,0]), 'Cap total rows per owner, keep public zero bin'
    assert np.array_equal(h,bounded_histogram(e.iloc[::-1],[10,20,30],2)), 'Stable row ID order'
    assert np.array_equal(bounded_histogram(e[e.driverId!=7],[10,20,30],2),[1,1,0])
    assert np.array_equal(bounded_histogram(e.iloc[:0],[10,20,30],2),[0,0,0])
    assert release_scale(5,.5,3)=={'scale':10.,'epsilon_total':1.5}
    bad_calls=[lambda:bounded_histogram(e,[10,10,20],2),lambda:bounded_histogram(e,[10],2),
        lambda:bounded_histogram(pd.concat([e,e.iloc[:1]]),[10,20],2),
        lambda:bounded_histogram(e,[10,20],0),lambda:bounded_histogram(e,[10,20],1.5),
        lambda:release_scale(1,0),lambda:release_scale(1,float('nan')),
        lambda:release_scale(1,1,0),lambda:release_scale(1,1,1.5),
        lambda:owned_counts({**db,'results':pd.DataFrame({'driverId':[999]})})]
    for call in bad_calls:
        try:call()
        except ValueError:pass
        else:raise AssertionError('Invalid privacy contract accepted')
    return 'PASS'


if __name__=='__main__':
    from relkit.privacy_l187 import owned_counts,bounded_histogram,release_scale
    print(checks(owned_counts,bounded_histogram,release_scale))
