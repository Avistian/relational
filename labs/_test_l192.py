"""Behavioral contracts; no model scores are used to teach selection."""
from relkit.setup_l192 import keyed_rows,available_history,select_candidate

def checks(keyed_rows,available_history,select_candidate):
    def reject(fn,*args):
        try:fn(*args)
        except (ValueError,TypeError):return
        raise AssertionError('Invalid input was accepted')
    rows=[{'entity':4,'cutoff':10,'label':1},{'entity':4,'cutoff':20,'label':0}]
    assert len(keyed_rows(rows))==2,'Entity alone loses repeated queries'
    reject(keyed_rows,rows+[rows[0]])
    reject(keyed_rows,[{'entity':True,'cutoff':10,'label':1}])
    reject(keyed_rows,[{'entity':4,'cutoff':float('nan'),'label':1}])
    h=[{'entity':4,'cutoff':1,'available_at':8,'label':1},
       {'entity':4,'cutoff':5,'available_at':15,'label':0},
       {'entity':4,'cutoff':10,'available_at':10,'label':1}]
    assert available_history(h,10)==[h[0]],'Require event before query and label available by query'
    assert available_history(h,15)==h,'Availability equality is admitted; query time must be later'
    reject(available_history,[dict(h[0],available_at=None)],10)
    reject(available_history,[dict(h[0],available_at=0)],10)
    expected=['d2-v2','d3-v2'];r=[dict(id=expected[0],split='val',auc=.6),dict(id=expected[1],split='val',auc=.7)]
    assert select_candidate(r,expected)=='d3-v2'
    assert select_candidate(list(reversed([dict(x,auc=.7) for x in r])),expected)=='d2-v2','Frozen order breaks ties'
    reject(select_candidate,r[:1],expected)
    reject(select_candidate,r+[r[0]],expected)
    reject(select_candidate,[dict(r[0],split='test'),r[1]],expected)
    for bad in [True,float('nan'),float('inf'),-.1,1.1]:reject(select_candidate,[dict(r[0],auc=bad),r[1]],expected)
    return {'status':'PASS','contracts':['complete query identity','label availability','complete validation-only selection']}

if __name__=='__main__':print(checks(keyed_rows,available_history,select_candidate))
