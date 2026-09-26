"""Behavioral contracts: temporal access, label maturity, paired comparisons."""
import json
from pathlib import Path
import numpy as np
from relkit.leakage_l104 import eligible_history, admissible_training, paired_ap, AuditFinder

def check():
    events=np.array([1.,3.,3.,5.]);available=np.array([1.,3.,4.,5.])
    np.testing.assert_array_equal(eligible_history(events,available,3.),[True,False,False,False])
    np.testing.assert_array_equal(eligible_history(events,available,4.),[True,True,True,False])
    np.testing.assert_array_equal(eligible_history(events,available,4.,inclusive=True),[True,True,True,False])
    np.testing.assert_array_equal(eligible_history(events,available,3.,inclusive=True),[True,True,False,False])
    assert not eligible_history(np.array([1.]),np.array([6.]),5.).any()
    np.testing.assert_array_equal(admissible_training(np.array([1,2,3,4]),np.array([3,5,6,7]),5),[True,True,False,False])
    base={'e':np.array([11,12]),'negative':np.array([7,8]),'p':np.array([.8,.6]),'n':np.array([.4,.2]),'batch':np.array([0,0])}
    assert paired_ap(base,base)['delta_pp']==0
    bad={**base,'negative':np.array([8,7])}
    try:paired_ap(base,bad)
    except ValueError:pass
    else:raise AssertionError('Changed negatives must reject comparison')
    d={'u':np.array([1,1,1]),'v':np.array([2,3,4]),'t':np.array([1.,3.,5.]),'e':np.array([1,2,3])}
    for mode,expected in [('strict',[1]),('inclusive',[1,2]),('lookahead',[1,2,3])]:
        f=AuditFinder(d,5,mode,lookahead=2)
        np.testing.assert_array_equal(f.find_before(1,3.)[:,1],expected)
        np.random.seed(4);ids,e,t=f.get_temporal_neighbor([1,0],[3.,3.],100)
        assert (ids[1]==0).all()
        assert f.audit['sampled_records']==100
        if mode=='strict':assert f.audit['nonpast_records']==0
        else:assert f.audit['nonpast_records']>0
    # Empty histories must consume the same uniform schedule as nonempty histories.
    f=AuditFinder(d,5,'strict');g=AuditFinder(d,5,'inclusive')
    np.random.seed(9);f.get_temporal_neighbor([1,0],[1.,3.],5);a=np.random.get_state()
    np.random.seed(9);g.get_temporal_neighbor([1,0],[1.,3.],5);b=np.random.get_state()
    np.testing.assert_array_equal(a[1],b[1]);assert a[2:]==b[2:]
    return {'status':'PASS','contracts':['strict and tied times','availability clock','label maturity','paired identity rejection','empty-history common draws','illegal sampled history detection']}
if __name__=='__main__':
    r=check();Path(__file__).with_name('_check_l104_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
