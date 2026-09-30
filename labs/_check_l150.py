"""Tests reject test-guided search, partial seed sets and conflated evidence gates."""
import math
from relkit.checkpoint_l150 import select_candidate, summarize_track, checkpoint_verdict

def rejects(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid evidence accepted')

def check_select(fn):
 rows=[dict(lr=.005,seed=100,epochs=10,complete=True,split='val',selection_mae=3.1),dict(lr=.001,seed=100,epochs=10,complete=True,split='val',selection_mae=3.1),dict(lr=.003,seed=100,epochs=10,complete=True,split='val',selection_mae=3.2)]
 assert fn(rows)==.001
 rejects(lambda:fn(rows[:-1]))
 rejects(lambda:fn([dict(r,split='test') for r in rows]))
 rejects(lambda:fn([dict(rows[0],complete=False)]+rows[1:]))
 rejects(lambda:fn([dict(rows[0],selection_mae=float('nan'))]+rows[1:]))
 rejects(lambda:fn([dict(rows[0],seed=0)]+rows[1:]))
 rejects(lambda:fn([dict(rows[0],test_mae=1.)]+rows[1:]))

def check_summary(fn):
 rows=[dict(seed=i,track='reference',complete=True,epochs=10,test_mae=3.8+i*.01) for i in range(5)]
 r=fn(rows,'reference',list(range(5)));assert abs(r['mean']-3.82)<1e-12 and r['sample_sd']>0
 rejects(lambda:fn(rows[:-1],'reference',list(range(5))))
 rejects(lambda:fn(rows+[rows[0]],'reference',list(range(5))))
 rejects(lambda:fn([dict(rows[0],track='replay')]+rows[1:],'reference',list(range(5))))
 rejects(lambda:fn([dict(rows[0],epochs=1)]+rows[1:],'reference',list(range(5))))
 rejects(lambda:fn([dict(rows[0],test_mae=float('inf'))]+rows[1:],'reference',list(range(5))))

def check_verdict(fn):
 r=fn(True,3.9,False,False);assert r['protocol']=='PASS' and r['paper_score']=='CLOSE' and r['competitive']=='NOT_ESTABLISHED' and r['learner']=='PENDING_WRITTEN_DEFENSE'
 r=fn(True,4.1,False,False);assert r['paper_score']=='OUTSIDE_TOLERANCE'
 r=fn(False,3.8,False,False);assert r['protocol']=='FAIL' and r['paper_score']=='INCOMPARABLE'
 r=fn(True,3.798,True,True);assert r['competitive']=='SUPPORTED_BY_SEPARATE_AUDIT' and r['learner']=='DEFENDED'
 rejects(lambda:fn(True,float('nan'),False,False))

if __name__=='__main__':
 check_select(select_candidate);check_summary(summarize_track);check_verdict(checkpoint_verdict);print('PASS: selection, full tracks, independent gates')
