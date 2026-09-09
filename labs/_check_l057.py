"""Contract tests: rows withheld, convex mixing, replacement, and no test-label input."""
import numpy as np
from relkit.cross_ensemble import train_indices, write_oof, blend, greedy_select

def checks():
    np.testing.assert_array_equal(train_indices(np.array([0,1,0,1]),0),[1,3])
    p=np.full((4,2),np.nan); seen=np.zeros(4,int)
    write_oof(p,seen,np.array([2,0]),np.array([[.2,.4],[.1,.3]]))
    np.testing.assert_allclose(p[[0,2]],[[.1,.3],[.2,.4]])
    try:write_oof(p,seen,np.array([0]),np.array([[.1,.3]]))
    except ValueError:pass
    else:raise AssertionError('Duplicate OOF coverage accepted')
    np.testing.assert_allclose(blend(np.array([[.2,.8],[.4,.6]]),[.75,.25]),[.35,.45])
    for w in ([-.1,1.1],[1,1],[np.nan,0]):
        try:blend(np.ones((2,2)),w)
        except ValueError:pass
        else:raise AssertionError('Nonconvex weights accepted')
    y=np.array([0,1]);p=np.array([[.1,.8],[.9,.2]])
    w,trace=greedy_select(p,y,steps=7)
    np.testing.assert_array_equal(w,[1,0]);assert len(trace)==7
    w,_=greedy_select(np.column_stack([p[:,0],p[:,0]]),y,steps=3)
    np.testing.assert_array_equal(w,[1,0])
    # Opposite overconfidence errors: the weaker single model earns weight.
    y=np.array([0,0,1,1]);p=np.array([[.01,.4],[.5,.01],[.99,.6],[.5,.99]])
    w,t=greedy_select(p,y,steps=20);assert (w>0).all()
    assert min(t[-1]['losses']) <= t[0]['chosen_loss']
    print('PASS OOF coverage, aligned rows, convex weights, deterministic replacement and complementary errors')
if __name__=='__main__':checks()
