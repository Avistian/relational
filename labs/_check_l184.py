"""Live learner contracts: temporal admission, Gaussian arithmetic and complete keys."""
import math
import numpy as np

def checks(sample, gaussian, score):
    edges=[(0,1),(0,2),(1,3),(2,4),(0,5)]
    times=[0,8,9,7,11,10]
    assert sample(edges,times,0,10,4)==[0,1,2,3], 'BFS must reject equal/future times'
    assert sample(edges,times,0,9,4)==[0,1,3], 'earlier cutoff changes legal context'
    assert sample(edges[::-1],times,0,10,4)==[0,1,2,3], 'stable tie order'
    x=gaussian(np.array([0.,2.,4.]),2.,2.)
    assert np.allclose(x,[math.exp(-1),1,math.exp(-1)]), 'source kernel has no factor 1/2'
    for width in [0,-1,float('nan')]:
        try:gaussian(np.array([0.]),0.,width)
        except ValueError:pass
        else:raise AssertionError('invalid width accepted')
    y=[(1,2,1.),(1,3,5.),(2,3,7.)];p=[(2,3,6.),(1,2,2.),(1,3,5.)]
    assert abs(score(y,p)-2/3)<1e-12, 'join on entity AND cutoff'
    for bad in [p[:-1],p+[p[0]],[(2,3,float('nan'))]+p[1:]]:
        try:score(y,bad)
        except ValueError:pass
        else:raise AssertionError('corrupt predictions accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.gelgt_l184 import temporal_bfs, gaussian_features, keyed_mae
    print(checks(temporal_bfs,gaussian_features,keyed_mae))
