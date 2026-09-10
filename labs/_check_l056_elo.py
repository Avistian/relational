"""Independent behavioral checks for the new L056 rating analysis."""
import numpy as np
import pandas as pd
from relkit.leaderboard_elo import paired_wins, fit_elo


def checks():
    ix=pd.MultiIndex.from_tuples([('a',0),('b',0),('b',1),('b',2)],names=['dataset','fold'])
    errors=pd.DataFrame([[0,1],[1,0],[1,0],[1,0]],index=ix,columns=['A','B'])
    w=paired_wins(errors)
    assert w.shape==(2,2,2)
    np.testing.assert_allclose(w.mean(0),[[.5,.5],[.5,.5]])
    np.testing.assert_allclose(fit_elo(w),[1000,1000],atol=1e-7)
    # A wins 3 of 4: analytical unpenalised gap = 400 log10(3).
    w=np.array([[[.5,.75],[.25,.5]]]*4)
    r=fit_elo(w)
    assert abs((r[0]-r[1])-400*np.log10(3))<.001
    np.testing.assert_allclose(fit_elo(w[:,::-1,::-1]),r[::-1],atol=1e-5)
    ties=pd.DataFrame([[1,1]],index=pd.MultiIndex.from_tuples([('a',0)],names=['dataset','fold']))
    np.testing.assert_allclose(paired_wins(ties),.5)
    assert np.isfinite(fit_elo(np.array([[[.5,1],[0,.5]]]))).all()
    for bad in (np.ones((2,2)),np.full((2,2,2),np.nan),np.ones((2,2,2))):
        try:fit_elo(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid win tensor accepted')
    print('PASS: dataset weighting, ties, analytical odds, permutation, complete separation and invalid tensors')

if __name__=='__main__':checks()
