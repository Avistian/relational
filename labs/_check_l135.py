"""Behavior contracts: completeness, validation-only selection, reservation and pairing."""
import json, math
from pathlib import Path

def check_select(fn):
    rows=[dict(config=c,seed=s,selection_mae=v) for c,vals in [('a',[2.,4.]),('b',[3.,2.])] for s,v in zip([100,101],vals)]
    assert fn(rows,['a','b'],[100,101])['winner']=='b'
    for bad in [rows[:-1],rows+[rows[0]],rows+[dict(config='c',seed=100,selection_mae=0.)], [dict(r,selection_mae=float('nan')) for r in rows]]:
        try:fn(bad,['a','b'],[100,101])
        except ValueError:pass
        else:raise AssertionError('Incomplete/duplicate/unplanned/nonfinite search accepted')
    tied=[dict(r,selection_mae=3.) for r in rows]
    assert fn(tied,['b','a'],[100,101])['winner']=='a'
    assert fn([dict(r,test_mae=-999 if r['config']=='a' else 999) for r in rows],['a','b'],[100,101])['winner']=='b'

def check_reserve(fn):
    assert math.isclose(fn([1.,2.],2,900,.00022572,3,10),3.406296)
    for args in [([6.9],1,900,.00022572,3,10),([],1,-1,1,0,10),([],1,1,float('nan'),0,10),([],0,1,1,0,10)]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid or over-budget dispatch accepted')

def check_paired(fn):
    r=fn([dict(seed=0,mae=4.),dict(seed=1,mae=5.)],[dict(seed=1,mae=4.),dict(seed=0,mae=4.5)])
    assert r['differences']==[.5,-1.] and r['mean_difference']==-.25
    for a,b in [([dict(seed=0,mae=1.)],[dict(seed=1,mae=1.)]),([dict(seed=0,mae=1.)]*2,[dict(seed=0,mae=1.)])]:
        try:fn(a,b)
        except ValueError:pass
        else:raise AssertionError('Mismatched/duplicate seeds accepted')

if __name__=='__main__':
    from relkit.tuning_l135 import select_configuration,reserve_budget,paired_differences
    for check,fn in [(check_select,select_configuration),(check_reserve,reserve_budget),(check_paired,paired_differences)]:check(fn)
    report=dict(status='PASS',contracts=3,negative_cases=10)
    Path(__file__).with_name('_check_l135_results.json').write_text(json.dumps(report,indent=2));print(report)
