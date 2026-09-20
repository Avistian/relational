"""Checkpoint behavior acceptance tests; written before implementation."""
import copy, importlib.util, json, sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
assert (P/'relkit/checkpoint_l100.py').exists(), 'L100 checkpoint implementation missing'
sys.path.insert(0,str(P/'relkit'))
import checkpoint_l100 as m
torch.set_num_threads(2)
# Hand-built graph: distinct local/global IDs, unequal relation degrees, isolated paper.
def fixture():
 gen=torch.Generator().manual_seed(100)
 edges=[torch.tensor([[0,1,2,3,3],[0,0,1,1,2]]),None,torch.tensor([[0,2,3],[0,1,0]]),None]
 edges[1]=edges[0].flip(0);edges[3]=edges[2].flip(0)
 return {'x':[torch.randn(5,3,generator=gen),torch.randn(3,1,generator=gen),torch.randn(2,1,generator=gen)],'edges':edges,'y':torch.tensor([0,1,2,0,1]),'train':torch.tensor([3,0,4]),'val':torch.tensor([1]),'test':torch.tensor([2])}
g=fixture();data=m.as_heterodata(g)
batch=next(iter(m.make_loader(data,[3,0],-1,2)))
for e in m.EDGE_TYPES:assert torch.equal(m.global_edges(batch,e),data[e].edge_index[:,batch[e].e_id])
z=torch.randn(batch['paper'].num_nodes,3,requires_grad=True)
a=m.seed_loss(z,batch);b=torch.nn.functional.cross_entropy(z[:2],batch['paper'].y[:2]);assert torch.equal(a,b)
a.backward();assert torch.equal(z.grad[2:],torch.zeros_like(z.grad[2:]))
changed=batch.clone();changed['paper'].y[2:]=(changed['paper'].y[2:]+1)%3
assert torch.equal(m.seed_loss(z,changed),a)
assert m.batch_weight(3,5)==.6 and m.batch_weight(2,5)==.4
for invalid in [(0,5),(6,5),(2,0)]:
 try:m.batch_weight(*invalid)
 except ValueError:pass
 else:raise AssertionError('invalid batch weight accepted')
reports=[]
for arm in m.ARMS:
 for bs in [1,2,5]:
  reports.append(m.audit_batches(g,arm,bs,width=8))
  assert reports[-1]['max_logit_gap']<2e-6
  assert reports[-1]['max_gradient_gap']<2e-6
 # Perturb test labels: training and validation selection must remain identical.
 altered=copy.deepcopy(g);altered['y'][g['test']]=1
 a,ma=m.fit(g,arm,.01,0,epochs=2,width=8,batch_size=2)
 b,mb=m.fit(altered,arm,.01,0,epochs=2,width=8,batch_size=2)
 assert a['trace']==b['trace'] and all(torch.equal(v,mb.state_dict()[k]) for k,v in ma.state_dict().items())
# Same initialization in common ablation tensors and identical training samples for all arms.
h=m.initialize(g,'hgt',0,8).state_dict();u=m.initialize(g,'hgt_uniform',0,8).state_dict()
assert all(torch.equal(v,h[k]) for k,v in u.items())
fingerprints=[m.fit(g,arm,.01,1,epochs=2,width=8,batch_size=2)[0]['sample_hashes'] for arm in m.ARMS]
assert all(fingerprints[0]==f for f in fingerprints)
# Empty auxiliary stores must remain valid, including an isolated seed.
empty=fixture();empty['edges']=[torch.empty(2,0,dtype=torch.long) for _ in range(4)]
for arm in m.ARMS:m.audit_batches(empty,arm,2,width=8)
report={'status':'PASS','typed_global_ids':'PASS','context_label_isolation':'PASS','heldout_label_intervention':'PASS','paired_sampling':'PASS','common_ablation_initialization':'PASS','empty_relations':'PASS','fixture_audits':reports}
(P/'_verify_l100_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
