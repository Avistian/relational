"""Behavioral contracts and independent library comparison for negative sampling."""
import json
import importlib.metadata as metadata
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from negative_l097 import proposal, draw_negatives, pairwise_loss, evaluate

def rejects(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Invalid sampling contract accepted')

blocked=np.array([[True,False,False,False],[False,True,True,False]])
q=proposal(blocked,np.array([1,8,0,1]),'uniform')
assert np.allclose(q,[[0,1/3,1/3,1/3],[.5,0,0,.5]])
qd=proposal(blocked,np.array([1,8,0,1]),'degree')
assert qd[0,1]>qd[0,3]>qd[0,2]>0 and qd[1,0]==qd[1,3]
rejects(lambda:proposal(np.ones((1,3),bool),np.ones(3),'uniform'))
rejects(lambda:proposal(blocked,np.ones(4),'typo'))
rng=np.random.default_rng(97);users=np.repeat([0,1],30000)
x=draw_negatives(users,q,rng)
assert not blocked[users,x].any() and set(x[users==1])=={0,3}
assert max(abs(np.bincount(x[:30000],minlength=4)[1:]/30000-1/3))<.02
assert np.array_equal(draw_negatives(users[:100],q,np.random.default_rng(7)),draw_negatives(users[:100],q,np.random.default_rng(7)))
# user 0 -> item 0 is legal when unobserved: typed IDs are separate.
assert draw_negatives(np.array([0]),np.array([[1.,0.]]),rng).item()==0
# A heldout positive is not consulted by this training-only sampler.
assert q[0,3]>0
s=torch.tensor([0.,2.],requires_grad=True);n=torch.tensor([0.,-1.],requires_grad=True)
loss=pairwise_loss(s,n);expected=(np.log(2)+np.log1p(np.exp(-3)))/2
assert abs(loss.item()-expected)<1e-7
loss.backward();assert (s.grad<0).all() and (n.grad>0).all()
# One relevant item at rank 2, seen item 0 masked even with score 100.
scores=np.array([[100.,2.,3.,0.]])
test=np.array([[0,1,5,0]])
result,records=evaluate(scores,np.array([[True,False,False,False]]),test,97)
assert np.isclose(result['full_ndcg'],1/np.log2(3)) and result['full_recall']==1
assert records[0]['full_top']==[2,1,3]
# Pinned PyG must return exactly the complement when all negatives requested.
from torch_geometric.utils import negative_sampling
edge=torch.tensor([[0,1],[1,0]])
neg=negative_sampling(edge,num_nodes=(2,3),num_neg_samples=6,method='dense')
assert set(map(tuple,neg.T.tolist()))=={(0,0),(0,2),(1,1),(1,2)}
report={'torch_geometric_version':metadata.version('torch-geometric'),'status':'PASS','contracts':['typed IDs','blocked pairs','uniform frequencies','degree support','empty pool rejected','invalid strategy rejected','seed replay','heldout not consulted','BPR value and gradient','ranking mask and independent metric','PyG exact complement']}
Path(__file__).with_name('_verify_l097_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
