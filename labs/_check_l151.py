"""Behavioral contracts shared with the learner notebook."""
import math,statistics

def rejects(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid evidence accepted')

def check_select(fn):
    rows=[dict(lr=lr,seed=100,epochs=20,complete=True,split='val',selection_auc=auc) for lr,auc in [(.00005,.70),(.0001,.71),(.0002,.71)]]
    assert fn(rows)==.0001,'Use validation maximum and lower-rate tie break'
    rejects(lambda:fn(rows[:2]));rejects(lambda:fn([dict(r,test_auc=.99) for r in rows]))
    rejects(lambda:fn([dict(r,complete=False) for r in rows]));rejects(lambda:fn([dict(r,selection_auc=float('nan')) for r in rows]))

def check_summary(fn):
    rows=[dict(seed=s,track='reference',complete=True,epochs=20,test_auc=.66+.01*s) for s in range(5)]
    r=fn(rows,'reference',list(range(5)));assert abs(r['mean']-.68)<1e-12
    assert abs(r['sample_sd']-math.sqrt(.00025))<1e-12
    rejects(lambda:fn(rows[:4],'reference',list(range(5))))
    rejects(lambda:fn(rows[:4]+[rows[0]],'reference',list(range(5))))
    rejects(lambda:fn([dict(r,track='search') for r in rows],'reference',list(range(5))))
    rejects(lambda:fn([dict(r,epochs=19) for r in rows],'reference',list(range(5))))

def check_entry(fn):
    r=fn('reference',True,.686,False)
    assert r['paper_score']=='CLOSE' and r['learner']=='PENDING_WRITTEN_DEFENSE'
    assert r['historical_identity']=='NOT_ESTABLISHED' and r['fresh_fe_comparison']=='NOT_RUN'
    assert fn('reference',True,.71,False)['paper_score']=='OUTSIDE_TOLERANCE'
    assert fn('selected',True,.686,False)['paper_score']=='INCOMPARABLE'
    assert fn('reference',False,.686,False)['paper_score']=='INCOMPARABLE'
    rejects(lambda:fn('reference',True,68.6,False))

if __name__=='__main__':
    from relkit.portfolio_l151 import select_candidate,summarize_track,portfolio_verdict
    check_select(select_candidate);check_summary(summarize_track);check_entry(portfolio_verdict)
    print('PASS: selection, complete-seed aggregation and claim boundaries')
