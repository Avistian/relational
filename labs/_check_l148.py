"""Behavioral contracts for live lesson functions."""
import numpy as np

def check_history(fn):
    # Distinct owners, exact boundary, future row, undated row, root exception.
    got=fn(np.array([100,1000]),np.array([90,89,99,995,1001,0]),np.array([0,0,0,1,1,0]),np.array([False,False,False,False,False,True]),np.array([False,False,False,False,False,False]),10)
    assert got.tolist()==[True,False,True,True,False,True],got
    assert fn(np.array([100]),np.array([1]),np.array([0]),np.array([False]),np.array([True]),10).tolist()==[True]
    assert not fn(np.array([100]),np.array([101]),np.array([0]),np.array([False]),np.array([True]),10)[0]

def check_keyed(fn):
    ref=[(7,10),(7,20)];predkeys=ref[::-1]
    assert fn(ref,[1,5],predkeys,[5,3])==1
    for keys,values in [([(7,10),(7,10)],[1,5]), ([(7,10)],[1]), (predkeys,[5,float('nan')])]:
        try:fn(ref,[1,5],keys,values)
        except ValueError:pass
        else:raise AssertionError('Must reject invalid keys/values')

def check_effect(fn):
    r=fn({0:1.,1:2.},{0:2.,1:4.},{0:3.,1:5.},{0:7.,1:10.})
    assert r['differences']==[3.,3.] and r['mean']==3 and r['sample_sd']==0
    try:fn({0:1,1:2},{0:1},{0:1,1:2},{0:1,1:2})
    except ValueError:pass
    else:raise AssertionError('Mismatched seeds accepted')

if __name__=='__main__':
    from relkit.ablation_l148 import history_mask,keyed_mae,interaction_summary
    check_history(history_mask);check_keyed(keyed_mae);check_effect(interaction_summary)
    print('PASS: owner/window boundaries, keyed metric, paired interaction')
