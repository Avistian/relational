"""Independent mechanism and information-boundary checks, written before implementation."""
import importlib.util,json,sys,copy
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
path=P/'relkit/checkpoint_l100.py'
assert path.exists(), 'L100 visible implementation is not yet present'
spec=importlib.util.spec_from_file_location('comparison',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)
x=torch.tensor([[2.],[8.],[-4.]],requires_grad=True);edge=torch.tensor([[0,1,2],[0,0,1]])
a=m.relation_mean(x,edge,3)
assert torch.equal(a,torch.tensor([[5.],[-4.],[0.]]))
a.sum().backward();assert torch.equal(x.grad,torch.tensor([[.5],[.5],[1.]]))
s=torch.tensor([[0.,1.],[0.,2.],[2.,3.]],dtype=torch.double,requires_grad=True);dst=torch.tensor([0,0,2])
a=m.receiver_softmax(s,dst,4);ref=torch.cat([s[:2].softmax(0),torch.ones_like(s[2:])])
assert torch.allclose(a,ref)
g1=torch.autograd.grad((a*a).sum(),s,retain_graph=True)[0];g2=torch.autograd.grad((ref*ref).sum(),s)[0];assert torch.allclose(g1,g2)
assert m.receiver_softmax(s[:0],dst[:0],4).shape==(0,2)
records=[{'arm':'hgt','lr':.01,'seed':0,'best_val_ce':.4},{'arm':'hgt','lr':.001,'seed':0,'best_val_ce':.2},{'arm':'hgt','lr':.01,'seed':1,'best_val_ce':.1},{'arm':'hgt','lr':.001,'seed':1,'best_val_ce':.2}]
assert m.choose_config(records,'hgt')==.001
# Dense oracle includes different relation degrees and an isolated receiver.
g={'x':[torch.randn(4,3),torch.randn(3,1),torch.randn(2,1)],'edges':[torch.tensor([[0,1,2],[0,0,2]]),torch.tensor([[0,0,2],[0,1,2]]),torch.tensor([[0,3],[0,1]]),torch.tensor([[0,1],[0,3]])], 'y':torch.tensor([0,1,2,1]),'train':torch.tensor([0,1]),'val':torch.tensor([2]),'test':torch.tensor([3])}
x=[torch.randn(len(t),8,dtype=torch.double,requires_grad=True) for t in g['x']]
layer=m.RGCNLayer(8).double();out=layer(x,g['edges']);dense=[layer.self_linear(t) for t in x]
for r,(source,target) in enumerate(m.ROUTES):
 e=g['edges'][r];mat=torch.zeros(len(x[target]),len(x[source]),dtype=torch.double)
 for i,j in e.T:mat[j,i]+=1
 mat=mat/mat.sum(1,keepdim=True).clamp_min(1);dense[target]=dense[target]+mat@x[source]@layer.weights[r]
for got,want in zip(out,dense):assert torch.allclose(got,want.relu(),atol=1e-10)
p=list(layer.parameters())+x
ga=torch.autograd.grad(sum(z.sum() for z in out),p,retain_graph=True);gb=torch.autograd.grad(sum(z.relu().sum() for z in dense),p)
assert all(torch.allclose(a,b,atol=1e-10) for a,b in zip(ga,gb))
# Exact output/gradient equivalence to the archived L093 modern-release HGT operator.
sys.path.insert(0,str(P/'relkit'));from hgt_l093 import HGTLayer as ReleasedHGT
for uniform in [False,True]:
 layer=m.HGTLayer(8,heads=2,uniform=uniform).double()
 out=layer(x,g['edges']);assert all(torch.isfinite(t).all() for t in out)
 if uniform:
  assert not any('q_linears' in n or 'relation_att' in n for n,_ in layer.named_parameters())
  continue
 old=ReleasedHGT(8,3,4,2,dropout=0,use_norm=True,use_rte=False).double();old.load_state_dict(layer.state_dict())
 offsets=[0,4,7];nt=torch.cat([torch.full((len(t),),i,dtype=torch.long) for i,t in enumerate(x)])
 ei=torch.cat([e+torch.tensor([[offsets[s]],[offsets[t]]]) for e,(s,t) in zip(g['edges'],m.ROUTES)],1)
 er=torch.cat([torch.full((e.shape[1],),r,dtype=torch.long) for r,e in enumerate(g['edges'])])
 expected=old(torch.cat(x),nt,ei,er,torch.zeros(len(er),dtype=torch.long))
 assert torch.allclose(torch.cat(out),expected,atol=1e-10)
 ga=torch.autograd.grad(torch.cat(out).square().sum(),list(layer.parameters())+x,retain_graph=True)
 gb=torch.autograd.grad(expected.square().sum(),list(old.parameters())+x)
 assert all(torch.allclose(a,b,atol=1e-9) for a,b in zip(ga,gb))
for arm in m.ARMS:
 model=m.Model([3,1,1],arm,width=8);z=model(g)
 assert z.shape==(4,3)
 # Edge ordering cannot alter the mathematical result.
 rev=copy.deepcopy(g);rev['edges']=[e.flip(1) for e in rev['edges']]
 assert torch.allclose(z,model(rev),atol=1e-6)
 # No test label may affect fitting, validation selection or chosen parameters.
 changed=copy.deepcopy(g);changed['y'][changed['test']]=0
 a,ma=m.fit(g,arm,.01,0,epochs=2,width=8);b,mb=m.fit(changed,arm,.01,0,epochs=2,width=8)
 assert a['trace']==b['trace'] and all(torch.equal(v,mb.state_dict()[k]) for k,v in ma.state_dict().items())
report={'status':'PASS','relation_dense_output_gradient':'PASS','softmax_dense_output_gradient':'PASS','hgt_l093_output_gradient_parity':'PASS','uniform_removes_score_parameters':'PASS','edge_order_invariance':'PASS','test_label_intervention_all_four_arms':'PASS','validation_only_config_selection':'PASS'}
(P/'_operator_l100_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
