"""Portable behavioral checks for the three live learner functions."""
def check165(context_fn, graph_fn, auc_fn):
    query={'entity':'u1','cutoff':10}
    context=[{'entity':'u1','cutoff':2,'label_end':5,'available_at':6,'label':1},
             {'entity':'u2','cutoff':8,'label_end':11,'available_at':11,'label':0},
             {'entity':'u3','cutoff':3,'label_end':7,'available_at':12,'label':1},
             {'entity':'u4','cutoff':4,'label_end':10,'available_at':10,'label':0},
             {'entity':'u1','cutoff':10,'label_end':10,'available_at':10,'label':1}]
    assert context_fn(context,query)==[context[0],context[3]], 'Maturity, arrival and query masking'
    rows=[{'id':'u1','event_at':None,'available_at':0}, {'id':'o1','event_at':5,'available_at':5},
          {'id':'p1','event_at':None,'available_at':8}, {'id':'o2','event_at':12,'available_at':12}]
    edges=[['u1','o1'],['o1','p1'],['u1','o2']]
    assert graph_fn(rows,edges,'u1',7,2)==['o1','u1'], 'Root cutoff must cover all hops'
    assert graph_fn(rows,edges,'u1',10,2)==['o1','p1','u1']
    assert graph_fn(rows,edges,'u1',10,0)==['u1']
    truth=[{'entity':'u1','cutoff':1,'label':1},{'entity':'u1','cutoff':2,'label':0},{'entity':'u2','cutoff':1,'label':1},{'entity':'u2','cutoff':2,'label':0}]
    predictions=[dict(entity=r['entity'],cutoff=r['cutoff'],score=s) for r,s in zip(truth,[.8,.8,.9,.1])]
    assert auc_fn(truth,list(reversed(predictions)))==.875, 'Complete keys and half credit for ties'
    for fn,args in [(context_fn,(context+[context[0]],query)),(context_fn,(context,dict(query,cutoff=True))),
                    (graph_fn,(rows,edges,'u1',10,-1)),(graph_fn,(rows,edges+[['u1','absent']],'u1',10,2)),
                    (auc_fn,(truth,predictions[:-1])),(auc_fn,(truth,predictions+[predictions[0]])),
                    (auc_fn,(truth,[dict(p,score=float('nan')) for p in predictions])),
                    (auc_fn,([dict(r,label=1) for r in truth],predictions))]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.context_l165 import eligible_context,visible_graph,keyed_auc
    print(check165(eligible_context,visible_graph,keyed_auc))
