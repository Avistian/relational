"""Behavioral checks for the corrected branch and live family-removal experiment."""
import numpy as np
from relkit import cross_experiment_v2 as ex

def checks():
    model=ex.TabM(3,k=4,width=8)
    assert model.blocks[0].S is None and model.blocks[0].bias.ndim==1
    y=np.array([0,0,1,1]);z=np.array([[.1,.8,.99],[.8,.1,.99],[.9,.2,.01],[.2,.9,.01]])
    a=ex.family_ablation(z,z,y,y,['A','B','C'],steps=8)
    b=ex.family_ablation(z,z,y,1-y,['A','B','C'],steps=8)
    for r,s in zip(a,b):
        assert r['weights']==s['weights'] and r['selected']==s['selected']
        assert r['weights'][r['removed_index']]==0
    calls=[];orig=ex.greedy_select
    def spy(p,y,steps):calls.append(p.shape[1]);return orig(p,y,steps)
    ex.greedy_select=spy
    try:ex.family_ablation(z,z,y,y,['A','B','C'],8)
    finally:ex.greedy_select=orig
    assert calls==[3,2,2,2],calls
    assert abs(a[2]['test_gap'])<1e-12
    print('PASS: corrected mini; ablation reselects; live helper invoked; test labels cannot change choices')
if __name__=='__main__':checks()
