"""Behavioral contracts for live learner functions and paid admission."""
import numpy as np

def checks(schedule, score, paired, reserve):
    rows=schedule(1100,[64,128,256,512,1024],'task:0')
    assert set(rows)=={64,128,256,512,1024}
    for k,idx in rows.items():
        assert len(idx)==k and len(set(idx))==k
        assert np.array_equal(idx,rows[1024][:k])
        assert min(idx)>=0 and max(idx)<1100
    assert not np.array_equal(rows[1024],schedule(1100,list(rows),'task:1')[1024])
    assert np.array_equal(rows[1024],schedule(1100,list(rows),'task:0')[1024])
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid evidence accepted')
    reject(lambda:schedule(20,[64],'task'))
    reject(lambda:schedule(20,[2,2],'task'))
    reject(lambda:schedule(20,[0],'task'))
    keys=np.array([[7,100],[7,200],[8,100],[8,200]])
    y=np.array([0,1,1,0]);p=np.array([.2,.9,.7,.7]);order=[2,0,3,1]
    assert score(keys,y,keys[order],p[order])==.875
    reject(lambda:score(keys,y,keys[[0,0,2,3]],p))
    reject(lambda:score(keys,y,keys[:3],p[:3]))
    reject(lambda:score(keys,y,keys,[0,1,float('nan'),.5]))
    reject(lambda:score(keys,y,keys,[0,1,2,.5]))
    a=[dict(context=k,seed=s,auc=.6+.01*s+(.02 if k==128 else 0)) for k in [64,128] for s in range(3)]
    result=paired(a,[64,128],[0,1,2])
    assert abs(result['doublings'][0]['mean_gain']-.02)<1e-12
    assert result==paired(list(reversed(a)),[64,128],[0,1,2])
    reject(lambda:paired(a[:-1],[64,128],[0,1,2]))
    reject(lambda:paired(a+[a[0]],[64,128],[0,1,2]))
    reject(lambda:paired(a+[dict(context=256,seed=0,auc=.9)],[64,128],[0,1,2]))
    budget=dict(cap_usd=10,planned_stop_usd=8,overhead_usd=3,reservations=[])
    b=reserve(budget,'pilot',600,.00028372)
    assert budget['reservations']==[] and b['reserved_usd']<8
    reject(lambda:reserve(b,'pilot',600,.00028372))
    reject(lambda:reserve(b,'huge',30000,.00028372))
    reject(lambda:reserve(b,'negative',-1,.00028372))
    return 'PASS'

if __name__=='__main__':
    import importlib.util
    from pathlib import Path
    assert importlib.util.find_spec('relkit.few_shot_l176'), 'L176 implementation is missing'
    from relkit.few_shot_l176 import nested_support,keyed_auc,paired_curve,reserve_cost
    print(checks(nested_support,keyed_auc,paired_curve,reserve_cost))
