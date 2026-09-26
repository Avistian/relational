"""Independent protocol checks against source plus adversarial availability fixtures."""
import sys,json,time,importlib
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'relkit'),str(P/'sources/l107/original')]
import snapshot_l107 as m,sbm_l107 as s,wiki_snapshot_l107 as w,taskers_utils as tu

def audit():
 torch.set_num_threads(1)
 # Original sampling returns the same candidate sequence and consumes the same random draws.
 pos=np.array([[0,1],[1,2],[2,0],[2,3]]);existing=np.array([0,1,2,3,0,1,2,3])
 np.random.seed(123);original=tu.get_non_existing_edges({'idx':torch.tensor(pos),'vals':torch.ones(4)},20,4,True,existing)['idx'].numpy();state=np.random.get_state()
 rng=np.random.RandomState(123);port=s.smart_negatives(pos,existing,4,rng,mult=5)
 np.testing.assert_array_equal(original,port);np.testing.assert_array_equal(state[1],rng.get_state()[1])
 # Native boundary, duplicate timestamp and empty intervals.
 for width in [1.,3.,10.]:
  ts=np.array([0,width-0.01,width,width,2*width+0.1])
  for q in [0,width-1e-6,width,width+1e-6,3*width]:
   expected=np.array([int(t//width) for t in ts if (int(t//width)+1)*width<=q])
   np.testing.assert_array_equal(m.completed_bins(ts,width,q),expected)
 ev={'t':np.array([0,1,1,1,2,3,3,4]),'split':np.zeros(8)}
 batches=list(w.time_batches(ev,0,2));assert [len(x) for x in batches]==[4,3,1]
 for left,right in zip(batches,batches[1:]):assert ev['t'][left[-1]]<ev['t'][right[0]]
 # MRR is average over ALL relevant ranks, not reciprocal first positive.
 pairs=np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
 y=np.array([0,1,1,0,0,0,0,0,0]);prob=np.array([.9,.8,.7,.6,.5,.4,.3,.2,.1])
 assert abs(s.sbm_metrics(pairs,y,prob,3)['mrr']-(1/2+1/3)/2)<1e-12
 # Node histories remain aligned: permuting every snapshot, state and edges commutes with GCN-GRU.
 torch.manual_seed(4);model=m.SnapshotGRU(4,3);x=torch.randn(5,4);h=torch.randn(5,3);a=torch.eye(5);perm=torch.tensor([2,4,1,0,3])
 torch.testing.assert_close(model.step(a,x,h)[perm],model.step(a[perm][:,perm],x[perm],h[perm]))
 # Match full versus chunked pair-head loss and gradients (one optimizer step, no detaching the GCN).
 d={'adj':[torch.eye(6).to_sparse()]*6,'features':[torch.randn(6,4) for _ in range(6)],'masks':[torch.zeros(6,1)]*6}
 torch.manual_seed(8);model=m.EvolveGCN(4,3,'H');head=m.PairClassifier(3,5);pairs=np.array([[i,j] for i in range(6) for j in range(6)]);labels=(pairs[:,0]==pairs[:,1]).astype(np.int64);weights=torch.tensor([.1,.9])
 torch.manual_seed(15);z=model(d['adj'],d['features'],d['masks']);loss=m.weighted_cross_entropy(head(z,torch.tensor(pairs)),torch.tensor(labels),weights);loss.backward();g=[p.grad.clone() if p.grad is not None else None for p in list(model.parameters())+list(head.parameters())]
 model.zero_grad();head.zero_grad();torch.manual_seed(15);_,v=s.pair_pass(model,head,d,5,pairs,labels,'cpu',True,weights,chunk=7)
 assert abs(v-float(loss.detach()))<1e-7
 for old,p in zip(g,list(model.parameters())+list(head.parameters())):
  if old is not None:torch.testing.assert_close(old,p.grad,atol=2e-7,rtol=2e-5)
 return {'status':'PASS','source_candidate_sequence_and_rng':'EXACT','completed_snapshot_boundary_cases':15,'tie_safe_event_batches':'PASS','source_MRR_definition':'PASS','node_permutation':'PASS','chunked_head_loss_gradients':'PASS'}
if __name__=='__main__':
 r=audit();(P/'_audit_l107_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
