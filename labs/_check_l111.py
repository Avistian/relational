"""Independent protocol checks: split overlap, invalid IDs and held-out-label isolation."""
import numpy as np
from relkit.ogb_contract_l111 import audit_splits, majority_predictions
s={'train':np.array([0,2]),'valid':np.array([1]),'test':np.array([3])}
assert audit_splits(s,4)=={'train':2,'valid':1,'test':1}
for bad in [dict(s,test=np.array([2])),dict(s,test=np.array([4])),dict(s,train=np.array([0,0])),dict(s,train=np.array([0.,2.]))]:
 try:audit_splits(bad,4)
 except ValueError:pass
 else:raise AssertionError('Invalid split accepted')
y=np.array([[2],[0],[1],[0]])
p=majority_predictions(y,s['train']);assert p.shape==(4,1) and (p==1).all()
y[[1,3]]=9;assert np.array_equal(p,majority_predictions(y,s['train']))
print('PASS: disjoint identities, range, integer IDs, duplicates, training-only majority and stable tie')
