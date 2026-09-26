"""Deterministic course diagnostics; reused verbatim in standalone notebooks."""
from relkit.debug_l116 import *
# NOTEBOOK_RUN
# PROVIDED: fixed inputs. Your live training_loss/train_step/seed_loss are used below.
torch.set_num_threads(1)
report={'scope':'SYNTHETIC_COURSE_DIAGNOSTICS','status':'PASS'}
# A: gradient present does not imply parameters move. Same model/RNG per arm.
x=torch.tensor([[1.,0.],[.8,.2],[.9,.1],[1.1,-.1],
                [0.,1.],[.2,.8],[.1,.9],[-.1,1.1]])
y=torch.tensor([0,0,0,0,1,1,1,1]);idx=torch.arange(8)
edge=torch.tensor([[i,j] for start in [0,4] for i in range(start,start+4) for j in range(start,start+4) if i!=j]).T
adj=normalized_adjacency(edge,8)
curves={}
for arm in ['broken','repaired']:
    torch.manual_seed(19);model=GCN(2,8,2,dropout=0);model.reset_parameters()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01);curve=[]
    for epoch in range(60):
        before=[p.detach().clone() for p in model.parameters()]
        loss=(missing_step if arm=='broken' else train_step)(model,optimizer,x,adj,y,idx)
        curve.append({'epoch':epoch+1,'loss':float(loss),'gradient_l2':vector_norm(p.grad for p in model.parameters()),'update_l2':vector_norm(p-b for p,b in zip(model.parameters(),before))})
    curves[arm]=curve
assert all(r['update_l2']==0 and r['gradient_l2']>0 for r in curves['broken'])
assert curves['repaired'][-1]['loss']<curves['repaired'][0]['loss']*.3
assert all(r['update_l2']>0 for r in curves['repaired'])
report['update_traces']=curves
# B: forward and backward are different measurements. This scalar chain isolates gain.
rows=[]
for gain in [.5,1.,2.]:
    for depth in [0,1,2,4,8,16]:
        u=torch.tensor(1.,dtype=torch.float64,requires_grad=True);h=u
        for _ in range(depth):h=gain*h
        h.backward();expected=gain**depth
        assert abs(float(u.grad)-expected)<1e-12
        rows.append({'gain':gain,'depth':depth,'input_gradient':float(u.grad)})
report['gradient_chain']=rows
# C: fixed graph, no learned W. Symmetric GCN converges in degree-corrected coordinates.
a=torch.tensor([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]],dtype=torch.float64)
d=a.sum(1);s=a/d.sqrt()[:,None]/d.sqrt()[None,:];h0=torch.tensor([2.,4.,8.],dtype=torch.float64)
h=h0.clone();rows=[]
for k in range(65):
    corrected=h/d.sqrt()
    if k in [0,1,2,4,8,16,32,64]:rows.append({'depth':k,'h':h.tolist(),'degree_corrected_variance':float(corrected.var(unbiased=False)),'raw_variance':float(h.var(unbiased=False))})
    h=s@h
assert rows[-1]['degree_corrected_variance']<1e-20 and rows[-1]['raw_variance']>.01
report['smoothing']=rows
# D: corrupt held-out labels; neither supervised loss nor its gradient may change.
logits=torch.tensor([[2.,-1.],[-2.,1.],[4.,-4.],[-3.,3.]],requires_grad=True)
yall=torch.tensor([1,1,1,0,0,1]);n_id=torch.tensor([4,1,5,0]);b=2
p=logits.log_softmax(1);clean=seed_loss(p,yall,n_id,b)
changed=yall.clone();changed[[0,2,3,5]]=1-changed[[0,2,3,5]]
assert torch.equal(clean,seed_loss(p,changed,n_id,b))
wrong=F.nll_loss(p[:b],yall[:b])
report['seed_mapping']={'n_id':n_id.tolist(),'seed_count':b,'labels':yall[n_id[:b]].tolist(),'correct_loss':float(clean.detach()),'wrong_global_index_loss':float(wrong.detach())}
# Full-graph label-isolation uses the live Task 1 function, not a prefilled substitute.
train=torch.tensor([0,1]);labels=torch.tensor([0,1,1,0]);changed=labels.clone();changed[2:]=1-changed[2:]
l1=training_loss(p,labels,train);l2=training_loss(p,changed,train)
torch.testing.assert_close(l1,l2,rtol=0,atol=0)
all_before=F.nll_loss(p,labels);all_after=F.nll_loss(p,changed)
report['label_probe']={'train_only_before':float(l1.detach()),'train_only_after':float(l2.detach()),'all_labels_before':float(all_before.detach()),'all_labels_after':float(all_after.detach())}
Path('l116-task-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['update_traces','gradient_chain','smoothing']},indent=2))
print('Fixed loss:',curves['repaired'][0]['loss'],'->',curves['repaired'][-1]['loss'])
