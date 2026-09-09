"""Behavioral contracts for the learner's leaderboard audit operations."""
import numpy as np
import pandas as pd
from relkit.leaderboard import rank_errors, aligned_errors, macro_ranks, bootstrap_gap

def checks():
    np.testing.assert_array_equal(rank_errors([.4,.2,.2,.9]), [3,1.5,1.5,4])
    np.testing.assert_array_equal(rank_errors([9,2,4]), [3,1,2])
    for bad in ([1,np.nan], [], [np.inf,1]):
        try: rank_errors(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid errors accepted')
    # Same dataset patterns, unequal repetitions: only the naive average moves.
    ix=pd.MultiIndex.from_tuples([('large',0),('small',0),('small',1),('small',2)],names=['dataset','fold'])
    r=pd.DataFrame([[1,2],[2,1],[2,1],[2,1]], index=ix,columns=['A','B'])
    np.testing.assert_allclose(macro_ranks(r).mean(),[1.5,1.5])
    assert r.mean()['A']==1.75
    gap=bootstrap_gap(np.array([0.,0.,0.]),seed=56,n_boot=200)
    np.testing.assert_array_equal(gap,[0,0,0])
    g=bootstrap_gap(np.array([-1.,0.,2.]),seed=56,n_boot=200)
    np.testing.assert_allclose(g,bootstrap_gap(np.array([-1.,0.,2.]),seed=56,n_boot=200))
    assert g[1]<=g[0]<=g[2]
    df=pd.DataFrame([dict(dataset='d',fold=0,arm=a,metric='rmse',metric_error=e,imputed=False) for a,e in [('A',2.),('B',1.)]])
    assert aligned_errors(df,['A','B']).iloc[0].tolist()==[2,1]
    for bad in (df.iloc[:1],pd.concat([df,df.iloc[:1]]),df.assign(imputed=True),df.assign(metric=['rmse','roc_auc'])):
        try:aligned_errors(bad,['A','B'])
        except ValueError:pass
        else:raise AssertionError('Missing, duplicate, imputed or mixed-metric row accepted')
    print('PASS: ties, invalid values, equal dataset weighting, reproducible cluster bootstrap, coverage guards')

if __name__=='__main__':checks()
