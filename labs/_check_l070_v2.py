"""Meaningful boundary tests for the live L070 checkpoint operations."""
import copy, json
from pathlib import Path
import numpy as np

def check(namespace=None):
    from relkit import checkpoint_l070_v2 as core
    s=vars(core) if namespace is None else namespace
    score,choose,panel,boot,cost,erase=[s[n] for n in ['binary_loss','choose_candidate','complete_panel','dataset_bootstrap','lifecycle_cost','erase_feature']]
    def rejects(fn):
        try:fn()
        except (ValueError,RuntimeError):return
        raise AssertionError('Invalid contract was accepted')
    assert abs(score([0,1],[.2,.8])+np.log(.8))<1e-12
    assert choose([.3,.2,.2])==1
    rejects(lambda:score([0,1],[.2,np.nan]));rejects(lambda:score([0,2],[.2,.8]))
    rejects(lambda:choose([]));rejects(lambda:choose([.1,np.nan]))
    rows=[dict(dataset=d,arm=a,seed=k,error=v+k*.01) for d,v in [('A',.1),('B',.5),('C',.9)] for a in ['x','m','z'] for k in [0,1,2]]
    result=panel(rows,['A','B','C'],['x','m','z'],[0,1,2])
    assert result['mean_ranks']==dict(x=2.,m=2.,z=2.)
    rejects(lambda:panel(rows[:-1],['A','B','C'],['x','m','z'],[0,1,2]))
    rejects(lambda:panel(rows[:9],['A','B','C'],['x','m','z'],[0,1,2]))
    rejects(lambda:panel(rows+rows[:1],['A','B','C'],['x','m','z'],[0,1,2]))
    b=boot(np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]),2000,70)
    assert b['units']==3 and abs(b['mean']-.5)<1e-12
    assert b==boot(np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]),2000,70)
    assert cost(100,1,30)==cost(10,4,30)==130
    rejects(lambda:cost(-1,1,2))
    train=np.array([[2.,5.],[4.,6.],[6.,7.]]);q=np.array([[1.,10.],[9.,20.]])
    np.testing.assert_array_equal(erase([[2],[3]],[[0],[9]],0),[[2.5],[2.5]])
    original=q.copy();np.testing.assert_array_equal(erase(train,q,0),[[4,10],[4,20]])
    np.testing.assert_array_equal(q,original);rejects(lambda:erase(train,q,2))
    # Changed helpers inside a generator, function defaults and closures must bind.
    ns={};exec('def hidden(x):return x+1\ndef outer():return sum(hidden(x) for x in [1,2])',ns)
    a=s['stable_code'](ns['outer']);exec('def hidden(x):return x+2',ns)
    assert a!=s['stable_code'](ns['outer'])
    ns={};exec('def h(x):return x+1\ndef outer(x,f=h):return f(x)',ns)
    json.dumps(s['stable_code'](float('nan')),allow_nan=False)
    a=s['stable_code'](ns['outer']);ns['outer'].__defaults__=(lambda x:x+2,)
    assert a!=s['stable_code'](ns['outer'])
    return dict(status='PASS',checks=['binary-loss','validation-ties','nonfinite-rejection','complete-roster','duplicate-rejection','dataset-bootstrap','lifecycle','frozen-feature-intervention','nested-helper','function-default'])

if __name__=='__main__':
    result=check();(Path(__file__).parent/'_check_l070_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
