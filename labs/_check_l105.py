"""Semantic contracts, not a model-score test. Run before the complete audit."""
import importlib.util,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_l105',P/'relkit/stream_l105.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(t,u=None,v=None,a=None):
 t=np.asarray(t,float);n=len(t)
 return dict(t=t,u=np.asarray(u if u is not None else [0]*n,dtype=np.int64),v=np.asarray(v if v is not None else [0]*n,dtype=np.int64),a=np.asarray(a if a is not None else t,float),e=np.arange(n,dtype=np.int64))

def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid contract accepted')

np.testing.assert_array_equal(m.bin_index(np.array([0.,9.999,10.,20.]),10),[0,0,1,2])
np.testing.assert_array_equal(m.bin_index(np.array([7.,17.]),10,origin=7),[0,1])
for width in [0,-1,np.nan,np.inf]:reject(lambda:m.bin_index(np.array([1.]),width))
reject(lambda:m.bin_index(np.array([np.nan]),10))
reject(lambda:m.bin_index(np.array([-1.]),10))
d=fixture([1,1,3,10],a=[1,1,12,10]);s=m.aggregate_events(d,10)
np.testing.assert_array_equal(s['count'],[3,1]);np.testing.assert_array_equal(s['release'],[12,20])
np.testing.assert_array_equal(m.visible_snapshot_mask(s,10),[False,False])
np.testing.assert_array_equal(m.visible_snapshot_mask(s,12),[True,False])
np.testing.assert_array_equal(m.visible_snapshot_mask(s,20),[True,True])
# An immutable complete window waits for ALL its constituent records.
s=m.aggregate_events(fixture([1,2],u=[0,1],a=[1,12]),10)
assert not m.visible_snapshot_mask(s,10).any()
assert m.visible_snapshot_mask(s,12).all()
assert len(m.aggregate_events(fixture([]),10)['count'])==0
# Bipartite identities and opposite endpoint pairs must remain distinct.
s=m.aggregate_events(fixture([1,2],u=[0,1],v=[1,0]),10);assert len(s['count'])==2
# Exact hand-computed witnesses: 2 strictly ordered pairs in bin zero;
# query-wise withheld past counts [0,0,2,0], nonpast exposures [3,3,1,1].
r=m.audit_stream(fixture([1,1,3,10]),10)
assert r['hidden_strict_pairs']==2 and r['collapsed_events']==2
assert r['withheld_past_sum']==2 and r['nonpast_exposure_sum']==8
assert r['tied_timestamp_pairs']==1
# Availability is not just event time; last completed partial bin cannot be released early.
assert not m.visible_snapshot_mask(m.aggregate_events(fixture([21]),10),29).any()
reject(lambda:m.aggregate_events(fixture([2,1]),10))
reject(lambda:m.aggregate_events(fixture([1,2],a=[1,1]),10))
reject(lambda:m.aggregate_events({**fixture([1,2]),'e':np.array([0,0])},10))
# Same aggregate, different existence of an increasing-time two-hop path.
a=fixture([1,2],u=[0,1],v=[1,2]);b=fixture([1,2],u=[1,0],v=[2,1])
sa=m.aggregate_events(a,10);sb=m.aggregate_events(b,10)
for k in ['bin','u','v','count']:np.testing.assert_array_equal(sa[k],sb[k])
assert a['t'][a['u']==0][0]<a['t'][a['u']==1][0]
assert not b['t'][b['u']==0][0]<b['t'][b['u']==1][0]
# Independent brute-force tests expose tie, endpoint, and release mistakes.
rng=np.random.default_rng(105)
for trial in range(100):
 n=int(rng.integers(0,30));t=np.sort(rng.integers(0,61,n)).astype(float)
 d=fixture(t,u=rng.integers(0,4,n),v=rng.integers(0,3,n));w=int(rng.choice([1,5,10,30]));s=m.aggregate_events(d,w);r=m.audit_stream(d,w)
 groups={}
 for u,v,tt in zip(d['u'],d['v'],t):
  start=0
  while tt>=start+w:start+=w
  key=(start//w,int(u),int(v));groups[key]=groups.get(key,0)+1
 assert [(int(b),int(u),int(v),int(c)) for b,u,v,c in zip(s['bin'],s['u'],s['v'],s['count'])]==[(*k,v) for k,v in sorted(groups.items())]
 hidden=sum(t[i]<t[j] and int(t[i]//w)==int(t[j]//w) for i in range(n) for j in range(i+1,n))
 assert r['hidden_strict_pairs']==hidden
 withheld=sum(sum(tt<q and (int(tt//w)+1)*w>q for tt in t) for q in t)
 nonpast=sum(sum(tt>=q and int(tt//w)==int(q//w) for tt in t) for q in t)
 assert r['withheld_past_sum']==withheld and r['nonpast_exposure_sum']==nonpast
 for q in [0,5,20,61]:
  visible=m.visible_snapshot_mask(s,q)
  assert int(s['count'][visible].sum())==sum((int(tt//w)+1)*w<=q for tt in t)
report={'status':'PASS','randomized_cases':100,'contracts':['half-open bins','typed endpoints','multiplicity','strict tie handling','delayed arrival','partial final bin','order collision','independent brute-force access counts'],'learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_check_l105_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
