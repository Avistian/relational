"""Scientific contracts, used unchanged by the portable notebook CHECK cells."""
import math

def rejects(fn,*args):
    try:fn(*args)
    except ValueError:return
    raise AssertionError('Invalid evidence accepted')

def check_align(fn):
    keys=[(4,100),(4,200),(9,100)]
    assert fn(keys,[keys[2],keys[0],keys[1]],[30.,10.,20.])==[10.,20.,30.]
    for bad,values in [([keys[0],keys[0],keys[2]],[1,2,3]),(keys[:2],[1,2]),(keys+[(8,1)],[1,2,3,4]),(keys,[1,float('nan'),3]),(keys,[1,2])]:rejects(fn,keys,bad,values)
    rejects(fn,[keys[0],keys[0]],keys[:2],[1,2])
    rejects(fn,[(None,100)],[(None,100)],[1])

def check_metric(fn):
    r=fn([0,2,4],[1,4,4],2.)
    assert r['mae']==1. and r['nmae']==.5 and r['count']==3
    assert fn([0,2,4],[1,4,4],4.)['nmae']==.25
    for y,p,s in [([],[],2),([1],[1,2],2),([1],[2],0),([1],[2],-1),([1],[2],float('nan')),([1],[float('inf')],2)]:rejects(fn,y,p,s)

def check_board(fn):
    assert fn({'b':.3,'a':.1},['a','b'])==.2
    for scores,tasks in [({'a':.1},['a','b']),({'a':.1,'b':.3,'c':.2},['a','b']),({'a':float('nan')},['a']),({'a':.1},['a','a']),({},[])]:rejects(fn,scores,tasks)

if __name__=='__main__':
    from relkit.leaderboard_l136 import align_predictions,regression_score,complete_board
    for test,fn in [(check_align,align_predictions),(check_metric,regression_score),(check_board,complete_board)]:test(fn)
    print('PASS: keyed identity, normalized metric, exact coverage and invalid evidence rejection')
