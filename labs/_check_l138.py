"""Contract checks include boundary, future leakage, ties and malformed inputs."""
import numpy as np

def check_target(fn):
    assert fn([-91],0)==(False,None)
    assert fn([0],0)==(True,1)
    assert fn([-1,91],0)==(True,0)
    assert fn([-1,92],0)==(True,1)
    assert fn([-92,1],0)==(False,None)
    assert fn([-90,-90,0],0)==(True,1)
    assert fn([0,1],0,1)==(True,0)
    for events,cut,h in [([np.nan],0,91),([1],np.inf,91),([1],0,0),([[1]],0,91)]:
        try:fn(events,cut,h)
        except ValueError:pass
        else:raise AssertionError('invalid target input accepted')

def check_paths(fn):
    edges=[('u','r0'),('r0','p'),('u','r1'),('r1','p2'),('p','r2')]
    times={'u':None,'r0':0,'p':None,'r1':1,'p2':None,'r2':2}
    assert fn(edges,times,'u',0,2)==['p','r0','u']
    assert fn(edges,times,'u',0,4)==['p','r0','u']
    assert fn(edges,times,'u',1,2)==['p','p2','r0','r1','u']
    assert fn(edges,times,'u',0,0)==['u']
    try:fn(edges,times,'r1',0,2)
    except ValueError:pass
    else:raise AssertionError('future root accepted')

def check_auc(fn):
    assert fn([0,1],[.5,.5])==.5
    assert fn([0,0,1,1],[0,1,1,2])==.875
    assert fn([0,1],[0,1])==1
    assert fn([0,1],[1,0])==0
    rng=np.random.default_rng(138)
    for _ in range(30):
        y=np.r_[0,1,rng.integers(0,2,28)];s=rng.integers(0,5,30)
        brute=np.mean([float(a>b)+.5*float(a==b) for a in s[y==1] for b in s[y==0]])
        assert abs(fn(y,s)-brute)<1e-12
    for y,s in [([1,1],[0,1]),([0,1],[1]),([0,1],[0,np.nan]),([0,2],[0,1])]:
        try:fn(y,s)
        except ValueError:pass
        else:raise AssertionError('invalid score input accepted')

if __name__=='__main__':
    from relkit.amazon_l138 import review_target,visible_paths,rank_auc
    for check,fn in [(check_target,review_target),(check_paths,visible_paths),(check_auc,rank_auc)]:check(fn)
    print('PASS: window endpoints, query cutoff, tied ranking and invalid inputs')
