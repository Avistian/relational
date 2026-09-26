"""Independent temporal eligibility and scalar/vector equality tests."""
import json
from pathlib import Path
import numpy as np
from relkit.sampling_l108 import window_bounds, choose_positions, expansion_size, TemporalIndex

def check():
    assert window_bounds(np.array([1.,3.,3.,8.]),3,2)==(0,1), 'Exclude every cutoff tie'
    assert window_bounds(np.array([1.,3.,3.,8.]),8,5)==(1,3), 'Include lower window endpoint'
    assert window_bounds(np.array([1.]),2,np.inf)==(0,1), 'Keep singleton history'
    assert expansion_size(2,20)==421
    assert expansion_size(2,5)==31
    assert choose_positions(3,5,'recent',np.zeros(5)).tolist()==[-1,-1,0,1,2]
    assert choose_positions(2,4,'uniform',np.array([.1,.9,.1,.9])).tolist()==[0,1,0,1]
    rng=np.random.default_rng(108); n=12; m=120
    events={'u':rng.integers(1,n,size=m),'v':rng.integers(1,n,size=m),'t':rng.integers(1,25,size=m).astype(float),'e':np.arange(1,m+1)}
    ix=TemporalIndex(events,n); cases=0
    for window in [1,5,np.inf]:
      for policy in ['uniform','recent']:
       for k in [1,5,20]:
        nodes=rng.integers(0,n,100);times=rng.integers(0,30,100).astype(float)
        draws=rng.random((100,k))
        fast=ix.sample(nodes,times,k,policy,window,draws)
        slow=ix.sample_scalar(nodes,times,k,policy,window,draws)
        for a,b in zip(fast,slow):np.testing.assert_array_equal(a,b)
        for row,(node,t) in enumerate(zip(nodes,times)):
          legal=[]
          for u,v,et,e in zip(events['u'],events['v'],events['t'],events['e']):
            if t-window<=et<t:
              if u==node:legal.append((v,e,et))
              if v==node:legal.append((u,e,et))
          legal.sort(key=lambda x:(x[2],x[1]))
          actual=[(int(a),int(b),float(c)) for a,b,c in zip(*(x[row] for x in fast)) if b]
          assert all(x in legal for x in actual), 'future/expired/wrong node edge'
          if policy=='recent':assert actual==legal[-k:]
          if legal and policy=='uniform':assert len(actual)==k
          if not legal:assert not actual
          cases+=1
    for args in [(np.array([1]),np.array([2.]),0,'uniform',1),(np.array([1]),np.array([2.]),2,'bad',1),(np.array([1]),np.array([2.]),2,'uniform',0)]:
      try:ix.sample(*args)
      except ValueError:pass
      else:raise AssertionError('Invalid config accepted')
    # Scalar task must control optimized boundaries as well as scalar reference.
    import relkit.sampling_l108 as module
    original=module.window_bounds
    module.window_bounds=lambda t,c,w:(np.searchsorted(t,c-w,'left'),np.searchsorted(t,c,'right'))
    try:
      changed=ix.sample_scalar([1],[3.],20,'recent',np.inf)
      correct=ix.sample([1],[3.],20,'recent',np.inf)
      assert not all(np.array_equal(a,b) for a,b in zip(changed,correct))
    finally:module.window_bounds=original
    return {'status':'PASS','independent_cases':cases,'policies':['uniform with replacement','recent'],'ties_singletons_empty_padding':'PASS'}
if __name__=='__main__':
 r=check();Path(__file__).with_name('_check_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
