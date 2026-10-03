"""Behavioral exam contracts; all notebook learner functions feed these tests."""
def checks(score, grid, gate):
    keys=[[1,10],[1,20],[2,10],[2,20]]
    assert score(keys,[0,1,0,1],keys[::-1],[.8,.5,.5,.1])==.875
    def rejects(fn,*args):
        try:fn(*args)
        except ValueError:return
        raise AssertionError('Invalid evidence accepted')
    rejects(score,keys,[0,1,0,1],keys[:-1],[.1,.2,.3])
    rejects(score,keys,[0,1,0,1],[keys[0]]*4,[.1,.2,.3,.4])
    rejects(score,keys,[0,1,0,1],keys,[.1,float('nan'),.3,.4])
    rejects(score,keys,[0,0,0,0],keys,[.1,.2,.3,.4])
    rejects(score,keys,[0,1,0,1],keys,[.1,1.2,.3,.4])
    rows=[dict(arm=a,seed=s,rows=702,support=512) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)]
    assert grid(rows)==dict(runs=30,predictions=21060)
    rejects(grid,rows[:-1]);rejects(grid,rows+[rows[0]])
    changed=[dict(r) for r in rows];changed[0]['rows']=701;rejects(grid,changed)
    changed=[dict(r) for r in rows];changed[0]['seed']=False;rejects(grid,changed)
    assert gate('PASS','PENDING','PENDING')==dict(state='INCOMPLETE',blockers=['proposal','defense'])
    assert gate('PASS','PASS','PASS')==dict(state='READY_FOR_TEACHER_REVIEW',blockers=[])
    assert gate('FAIL','PASS','PASS')['state']=='INCOMPLETE'
    rejects(gate,'COMPLETE','PASS','PASS')
    return 'PASS'
if __name__=='__main__':
    from relkit.exit_l200 import keyed_auc,complete_grid,exit_gate
    print(checks(keyed_auc,complete_grid,exit_gate))
