"""Behavioral contracts also executed against student and deliberately wrong functions."""
import copy
import numpy as np
from relkit.comparison_b23 import admit_grid,paired_comparison,claim_status
ARMS=['RDBPFN','RDBPFN_single','TabICLv1.1','Logistic']
def rejects(fn,*args):
 try:fn(*args)
 except (ValueError,TypeError):return
 raise AssertionError('Invalid evidence accepted')
def checks(admit,paired,claim):
 rows=[dict(arm=a,seed=s,rows=702,support=512,auc=.5+.01*s+.001*i) for i,a in enumerate(ARMS) for s in range(10)]
 assert admit(rows)==dict(runs=40,predictions=28080,paper_runs=30,course_runs=10)
 for bad in [rows[:-1],rows+[rows[0]],rows[1:]+[rows[0]|{'seed':True}],rows[1:]+[rows[0]|{'rows':701}],rows[1:]+[rows[0]|{'auc':float('nan')}],rows[1:]+[rows[0]|{'auc':1.1}]]:rejects(admit,bad)
 shuffled=list(reversed(rows));result=paired(shuffled,'RDBPFN','Logistic')
 np.testing.assert_allclose(result['per_seed'],np.full(10,.003),atol=1e-15)
 assert result['positive']==10 and result['seeds']==list(range(10))
 rejects(paired,rows[:-1],'RDBPFN','Logistic');rejects(paired,rows,'RDBPFN','RDBPFN')
 assert claim(True,True,False,False)==dict(numerical='CLOSE',historical='NOT_ESTABLISHED',deployment_validity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')
 assert claim(False,True,True,True)['numerical']=='INCOMPLETE'
 assert claim(True,False,True,False)['numerical']=='OUTSIDE_TOLERANCE'
 rejects(claim,'yes',True,False,False)
 return 'PASS'
if __name__=='__main__':print(checks(admit_grid,paired_comparison,claim_status))
