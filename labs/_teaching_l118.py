"""Portable synthetic task, explicitly separate from real-data/paper evidence."""
# NOTEBOOK_RUN
# The graph constructor and network call all three live student functions.
torch.set_num_threads(1)
torch.manual_seed(17)
stored=[(1,0),(0,2),(3,2),(4,3),(5,1)]
selected,chosen_edges=rdb_to_graph(6,stored,0)
assert selected==[0,1,2,5]
assert undirected_hops(6,stored,0)==[0,1,2,3,5]
# Four graph copies; connected row values determine synthetic labels.
features={'A':(torch.empty(4,0,dtype=torch.long),torch.tensor([[0.,0.]]*4)),
          'B':(torch.empty(8,0,dtype=torch.long),torch.tensor([[-1.,0.],[-.8,0.],[1.,0.],[.8,0.],[-.7,0.],[-1.2,0.],[.9,0.],[1.1,0.]]))}
types=torch.tensor([0,1,1]*4)
batch=torch.arange(4).repeat_interleave(3)
toy_nodes,toy_edges=rdb_to_graph(3,[(1,0),(2,0)],0)
assert toy_nodes==[0,1,2]
edge=torch.cat([computation_edges(len(toy_nodes),toy_edges)+3*i for i in range(4)],dim=1)
y=torch.tensor([0,1,0,1])
model=CvitkovicGCN({'A':([],2),'B':([],2)},{'A':0,'B':1},hidden=8,dropout=0)
optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
with torch.no_grad():before=float(F.cross_entropy(model(features,types,edge,batch,4),y))
for step in range(60):
 optimizer.zero_grad();loss=F.cross_entropy(model(features,types,edge,batch,4),y);loss.backward();optimizer.step()
with torch.no_grad():after=float(F.cross_entropy(model(features,types,edge,batch,4),y))
assert after<before/2,'A successful fit must actually reduce the loss.'
report={'status':'PASS','evidence':'COURSE_ONLY synthetic four-graph fit','selected_nodes':selected,'two_hop_nodes':undirected_hops(6,stored,0),'loss_before':before,'loss_after':after,'full_paper_run':'NOT_RUN in this notebook','learner':'PENDING_WRITTEN_DEFENSE'}
Path('l118-task-report.json').write_text(json.dumps(report,indent=2));print(report)
