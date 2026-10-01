"""Behavioral contracts also embedded in the portable notebook."""
def check162(reach, budget, verdict):
    nodes={'m1':'moons','m2':'moons','p1':'planets','s1':'stars','z1':'isolated'}
    edges=[['m1','p1'],['m2','p1'],['p1','s1']]
    assert reach(nodes,edges,'m1',0)==['m1']
    assert reach(nodes,edges,'m1',1)==['m1','p1']
    assert reach(nodes,edges,'m1',2)==['m1','m2','p1','s1']
    assert reach(nodes,edges,'m1',8)==['m1','m2','p1','s1']
    assert reach(nodes,edges,'m1',2,['moons','stars'])==['m1']
    assert reach(nodes,edges,'s1',1)==['p1','s1']
    assert budget([3,5],4)==dict(rows=2,total_tokens=8,retained_tokens=7,dropped_tokens=1,overflow_rows=[1],whole_table_pairs=64,row_pairs=34,retained_row_pairs=25)
    assert budget([3,5],8)['row_pairs']==34
    assert budget([0],4)['retained_tokens']==0
    base=dict(multitable=False,heldout_database=False,pretraining_audited=False,adaptation_legal=False,matched_baseline=False,predictions_verified=False)
    assert verdict(base)==dict(supported='RECONSTRUCTION_ONLY',transfer='NOT_ESTABLISHED',missing=['MULTITABLE_EVALUATION','HELD_OUT_DATABASE','PRETRAINING_PROVENANCE','LEGAL_ADAPTATION','MATCHED_BASELINE','VERIFIED_PREDICTIONS'])
    assert verdict(dict.fromkeys(base,True))['supported']=='TRANSFER_REVIEW_ELIGIBLE'
    partial=dict.fromkeys(base,True);partial['predictions_verified']=False
    assert verdict(partial)['supported']=='MULTITABLE_ONLY'
    for args in [(nodes,edges,'missing',1),(nodes,edges,'m1',True),(nodes,edges,'m1',-1),(nodes,[['m1','bad']],'m1',1),(nodes,edges,'m1',2,['stars']),(nodes,edges,'m1',2,'moons'),({' m1':'moons'},[],' m1',1),(nodes,[['m1','p1'],['p1','m1']],'m1',1)]:
        try:reach(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid graph accepted')
    for lengths,limit in [([],4),([True],4),([-1],4),([3],0),([3],True),([1.5],4)]:
        try:budget(lengths,limit)
        except ValueError:pass
        else:raise AssertionError('Invalid token input accepted')
    for record in [{},dict(base,multitable=1),dict(base,extra=True)]:
        try:verdict(record)
        except ValueError:pass
        else:raise AssertionError('Invalid evidence record accepted')
    assert nodes=={'m1':'moons','m2':'moons','p1':'planets','s1':'stars','z1':'isolated'}
    return 'PASS'

if __name__=='__main__':
    from relkit.vision_l162 import reachable_rows,token_budget,evidence_verdict
    print(check162(reachable_rows,token_budget,evidence_verdict))
