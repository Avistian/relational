"""Semantic contracts; fixtures deliberately expose leakage and alignment errors."""
import json,math
from pathlib import Path
from relkit.manual_fe_l129 import past_summary,align_predictions,choose_trial

def rejected(fn,*args):
    try:fn(*args)
    except ValueError:return
    raise AssertionError('Invalid input accepted')

def check_past(fn):
    # Event, arrival and value. Same-cutoff races excluded; arrival exactly t allowed.
    rows=[dict(entity=7,event=2,available=2,value=6),dict(entity=7,event=5,available=5,value=4),dict(entity=7,event=9,available=10,value=2),dict(entity=7,event=8,available=11,value=99),dict(entity=7,event=10,available=10,value=88),dict(entity=8,event=8,available=8,value=77)]
    assert fn(rows,7,10,8)==dict(count=2,mean=3.0,days_since_latest=1)
    assert fn(rows,9,10,8)==dict(count=0,mean=None,days_since_latest=None)
    assert fn(rows,7,10,1)==dict(count=0,mean=None,days_since_latest=None)
    future=rows+[dict(entity=7,event=12,available=12,value=-1000)]
    assert fn(rows,7,10,8)==fn(future,7,10,8)
    rejected(fn,rows,7,10,0)

def check_align(fn):
    # Same entity at two cutoffs proves entity-only lookup is wrong.
    keys=[(7,20),(8,10),(7,10)];pred=[2.,8.,1.];q=[(7,10),(7,20),(8,10)]
    assert fn(keys,pred,q)==[1.,2.,8.]
    rejected(fn,keys,[2,8],q)
    rejected(fn,keys,pred,[(7,10)])
    rejected(fn,[(7,10),(7,10)],[1,2],[(7,10),(8,10)])
    rejected(fn,keys,[2,float('nan'),1],q)
    rejected(fn,keys,pred,[(7,10),(7,10),(8,10)])

def check_select(fn):
    trials=[dict(number=0,val_mae=3.2,test_mae=9),dict(number=1,val_mae=3.0,test_mae=8),dict(number=2,val_mae=3.0,test_mae=1)]
    assert fn(trials)==1
    assert fn(list(reversed(trials)))==1
    for row in trials:row['test_mae']=-row['test_mae']
    assert fn(trials)==1
    rejected(fn,[]);rejected(fn,[dict(number=0,val_mae=float('nan'))])
    rejected(fn,[dict(number=0,val_mae=1),dict(number=0,val_mae=2)])

if __name__=='__main__':
    for fn,check in [(past_summary,check_past),(align_predictions,check_align),(choose_trial,check_select)]:check(fn)
    r=dict(status='PASS',contracts=['strict past plus arrival cutoff','complete unique query identity','first validation minimum independent of test'])
    (Path(__file__).parent/'_check_l129_results.json').write_text(json.dumps(r,indent=2));print(r)
