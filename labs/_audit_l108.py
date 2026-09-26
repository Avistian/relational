"""Model-path, source-input and changed-helper rejection evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from relkit.sampling_l108 import TemporalIndex
from relkit.tgat_l103 import TGAT
from _run_l108 import fingerprint
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(108)
events={'u':np.array([1,2,2,1,3]),'v':np.array([2,3,4,4,4]),'t':np.array([2.,3.,6.,7.,9.]),'e':np.arange(1,6)}
index=TemporalIndex(events,5)
class Scalar(TemporalIndex):
 def get_temporal_neighbor(self,nodes,cutoffs,num_neighbors=20):
  return self.sample_scalar(nodes,cutoffs,num_neighbors,self.policy,self.window)
scalar=Scalar(events,5)
model=TGAT(index,np.zeros((5,4),np.float32),np.random.default_rng(0).normal(size=(6,4)).astype(np.float32),dropout=0).eval()
args=(np.array([1,2]),np.array([3,4]),np.array([4,3]),np.array([8.,10.]),3)
with torch.no_grad():
 np.random.seed(108);a=model.contrast(*args);model.ngh_finder=scalar;np.random.seed(108);b=model.contrast(*args)
for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
# Trace recursive cutoffs: third call is sampled children at their edge times.
class Trace(TemporalIndex):
 def __init__(self,*a):super().__init__(*a);self.trace=[]
 def get_temporal_neighbor(self,n,t,num_neighbors=20):
  result=super().get_temporal_neighbor(n,t,num_neighbors);self.trace.append((np.array(n),np.array(t),result));return result
trace=Trace(events,5);model.ngh_finder=trace
with torch.no_grad():model.tem_conv(np.array([1]),np.array([8.]),2,3)
assert len(trace.trace)==3
np.testing.assert_array_equal(trace.trace[2][0],trace.trace[1][2][0].ravel())
np.testing.assert_array_equal(trace.trace[2][1],trace.trace[1][2][2].ravel())
assert not np.all(trace.trace[2][1]==8), 'Root cutoff incorrectly reused'
assert trace.audit['nonpast_records']==0
# Fingerprint transitive helpers in an isolated copy, never mutate running source.
import shutil,tempfile
import _run_l108 as runner
h=fingerprint()
with tempfile.TemporaryDirectory(prefix='l108-resume-') as tmp:
 root=Path(tmp)
 for name in runner.FILES:
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,target)
 original_p=runner.P;runner.P=root
 try:
  assert runner.fingerprint()==h
  helper=root/'relkit/sampling_l108.py'
  helper.write_text(helper.read_text().replace("side='left'", "side='right'"))
  assert runner.fingerprint()!=h
 finally:runner.P=original_p
assert fingerprint()==h
r={'status':'PASS','full_model_scalar_vector_prediction_parity':'EXACT','recursive_edge_cutoff_trace':'PASS','fingerprint_changed_helper_rejection':'PASS','fingerprint_restored':h}
(P/'_audit_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
