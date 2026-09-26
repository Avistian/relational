"""Small executable task readouts; no dataset or paper parity claims."""
from relkit.patterns_l115 import *
# NOTEBOOK_RUN
# RUN: your live graph_forward, pair_features and graph_mean are all used here.
torch.set_num_threads(1);torch.manual_seed(31)
# Node task: two disconnected groups, fixed synthetic attributes and labels.
x=torch.tensor([[1.,0.],[1.1,.1],[.9,-.1],[-1.,0.],[-1.1,.1],[-.9,-.1]])
edges=torch.tensor([[0,1,2,3,4,5],[1,2,0,4,5,3]])
adj=normalized_adjacency(edges,6);y=torch.tensor([0,0,0,1,1,1])
model=GCN(2,8,2,dropout=0.);opt=torch.optim.Adam(model.parameters(),lr=.03)
losses=[]
for step in range(60):
    model.train();opt.zero_grad();loss=F.nll_loss(model(x,adj),y);loss.backward();opt.step();losses.append(float(loss.detach()))
model.eval();node_pred=model(x,adj).argmax(1)
assert losses[-1]<losses[0] and torch.equal(node_pred,y)
# Link readout task: fixed Z and explicit directed candidate labels.
# These pairs are supervised examples, not edges fed into a message graph.
z=torch.tensor([[1.,2.],[3.,5.],[7.,11.]])
pairs=torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]])
link_y=torch.tensor([1.,0.,1.,0.,1.,0.]) # later-index destination, illustrative rule
features=pair_features(z,pairs,True);link_head=nn.Linear(4,1)
opt=torch.optim.Adam(link_head.parameters(),lr=.08);link_losses=[]
for step in range(120):
    opt.zero_grad();score=link_head(features).flatten();loss=F.binary_cross_entropy_with_logits(score,link_y);loss.backward();opt.step();link_losses.append(float(loss.detach()))
assert link_losses[-1]<link_losses[0]
assert torch.equal((link_head(features).flatten()>0).long(),link_y.long())
# Graph readout task: batch IDs preserve two differently sized examples.
batch=torch.tensor([0,1,0]);pooled=graph_mean(z,batch,2)
graph_y=torch.tensor([1,0]);graph_head=nn.Linear(2,2);opt=torch.optim.Adam(graph_head.parameters(),lr=.08);graph_losses=[]
for step in range(160):
    opt.zero_grad();loss=F.cross_entropy(graph_head(pooled),graph_y);loss.backward();opt.step();graph_losses.append(float(loss.detach()))
assert graph_losses[-1]<graph_losses[0] and torch.equal(graph_head(pooled).argmax(1),graph_y)
report={'status':'PASS','scope':'SYNTHETIC_COURSE_EXAMPLES; fit and score same tiny examples, no held-out generalization claim','node':{'shape':list(model(x,adj).shape),'loss_first':losses[0],'loss_last':losses[-1]},'link':{'shape':list(features.shape),'loss_first':link_losses[0],'loss_last':link_losses[-1],'labels':'Explicit illustrative candidate labels; not a benchmark negative sampler'},'graph':{'pooled':pooled.tolist(),'loss_first':graph_losses[0],'loss_last':graph_losses[-1]},'learner_status':'PENDING_WRITTEN_DEFENSE'}
Path('l115-task-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
