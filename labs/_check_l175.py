"""Behavioral contracts; synthetic fixtures are not performance evidence."""
import numpy as np
from relkit.zero_shot_l175 import aligned_auc, exposure_claim, audit_context, reserve_cost

def checks():
 keys=np.array([[7,100],[7,200],[8,100],[8,200]])
 y=np.array([0,1,1,0]);p=np.array([.2,.9,.7,.7]);order=[2,0,3,1]
 assert aligned_auc(keys,y,keys[order],p[order])==.875
 for bad in [keys[:3],np.array([[7,100],[7,100],[8,100],[8,200]])]:
  try:aligned_auc(keys,y,bad,p[:len(bad)])
  except ValueError:pass
  else:raise AssertionError('missing or duplicate key accepted')
 assert exposure_claim(False,True,True,False)=='NO_TARGET_GRADIENTS_WITH_LABEL_ACCESS'
 assert exposure_claim(False,False,False,False)=='STRICT_NO_TARGET_LABEL_ACCESS'
 assert exposure_claim(True,False,False,False)=='TARGET_TRAINED'
 assert exposure_claim(False,False,False,True)=='INVALID_TEST_EXPOSURE'
 result=audit_context([100,100,90,110,-2147483648],[False,False,False,False,False],[True,False,False,False,False],[True,True,True,False,False],[True,False,False,False,False],100,30)
 assert result['unmasked_query_targets']==0
 assert result['same_time_labels']==1 and result['unavailable_labels']==2 and result['future_cells']==1
 assert reserve_cost([],600,.00028372,2,8)<8
 try:reserve_cost([6.5],600,.00028372,2,8)
 except ValueError:pass
 else:raise AssertionError('budget exceeded')
 return {'status':'PASS','contracts':['composite keys and tied AUROC','exposure axes','context availability','aggregate reservation']}
def mutation_checks():
    wrongs={'aligned_auc':lambda *a: .5,
            'exposure_claim':lambda *a: 'STRICT_NO_TARGET_LABEL_ACCESS',
            'audit_context':lambda *a: {'unmasked_query_targets':0,'same_time_labels':0,'unavailable_labels':0,'future_cells':0}}
    rejected=[]
    for name,wrong in wrongs.items():
        original=globals()[name];globals()[name]=wrong
        try:
            try:checks()
            except AssertionError:rejected.append(name)
            else:raise RuntimeError('Wrong learner implementation passed: '+name)
        finally:globals()[name]=original
    assert len(rejected)==3
    return rejected
if __name__=='__main__':print(checks(),mutation_checks())
