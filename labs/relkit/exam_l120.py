"""Year 3 exam: explicit PyG query graphs and a small trainable typed model.
Course fixture only. The paper model/trainer remains separately pinned in rdl_l117.
"""
# %% Imports and immutable course fixture (PROVIDED)
import json
import torch
from torch_geometric.data import HeteroData, Batch
PERSON_KEYS=[90,10,70]
EVENT_PERSON=[10,90,90,10,70,90]
EVENT_MERCHANT=[8,8,4,4,8,4]
EVENT_TIME=[2,3,5,9,4,11]
EVENT_AVAILABLE=[2,3,8,9,4,11]
AMOUNTS=[1.,2.,4.,8.,3.,16.]
MERCHANT_KEYS=[8,4]

# %% Task 1: relational identities

def key_edges(primary_keys, foreign_keys):
    lookup={key:i for i,key in enumerate(primary_keys)}
    if len(lookup)!=len(primary_keys):raise ValueError('Duplicate primary key')
    pairs=[]
    for row,key in enumerate(foreign_keys):
        if key is None:continue
        if key not in lookup:raise ValueError('Dangling foreign key')
        pairs.append((row,lookup[key]))
    return torch.tensor(pairs,dtype=torch.long).reshape(-1,2).t().contiguous()

# %% Task 2: point-in-time visibility

def visible_rows(event_time, available_time, cutoff):
    return (torch.as_tensor(event_time)<=cutoff)&(torch.as_tensor(available_time)<=cutoff)

# %% Query-specific temporal graph extraction (PROVIDED)

def query_graph(person_key, cutoff, hops=2):
    person=PERSON_KEYS.index(person_key)
    ep=key_edges(PERSON_KEYS,EVENT_PERSON);em=key_edges(MERCHANT_KEYS,EVENT_MERCHANT)
    legal=visible_rows(EVENT_TIME,EVENT_AVAILABLE,cutoff)
    full={('event','owner','person'):ep[:,legal],('event','shop','merchant'):em[:,legal]}
    full.update({(dst,'rev_'+rel,src):edges.flip(0) for (src,rel,dst),edges in list(full.items())})
    # The root cutoff was applied before traversal and never becomes a neighbor time.
    reached={('person',person)};frontier=set(reached)
    for _ in range(hops):
        found={(dst,int(b)) for (src,rel,dst),edges in full.items() for a,b in edges.t().tolist() if (src,int(a)) in frontier}
        frontier=found-reached;reached|=found
    ids={kind:sorted(i for k,i in reached if k==kind) for kind in ['person','event','merchant']}
    graph=HeteroData()
    features={'person':torch.tensor([[1.,0.]]*3),'merchant':torch.tensor([[1.,0.]]*2),
              'event':torch.tensor([[a,(cutoff-t)/10.] for a,t in zip(AMOUNTS,EVENT_TIME)])}
    maps={k:{global_id:local for local,global_id in enumerate(v)} for k,v in ids.items()}
    for kind,index in ids.items():
        graph[kind].x=features[kind][index];graph[kind].n_id=torch.tensor(index,dtype=torch.long)
    graph['person'].root_mask=torch.tensor([i==person for i in ids['person']],dtype=torch.bool)
    for (src,rel,dst),edges in full.items():
        pairs=[(maps[src][a],maps[dst][b]) for a,b in edges.t().tolist() if a in maps[src] and b in maps[dst]]
        graph[src,rel,dst].edge_index=torch.tensor(pairs,dtype=torch.long).reshape(-1,2).t().contiguous()
    graph.query_key=torch.tensor([[person_key,cutoff]])
    return graph

# %% Task 3: mini-batch seed identity

def seed_positions(batch):
    return batch['person'].root_mask.nonzero(as_tuple=False).flatten()

# %% Task 4: mature seed-only supervision

def seed_loss(predictions, roots, targets, label_available, fit_cutoff):
    if bool((label_available>fit_cutoff).any()):raise ValueError('Immature training label')
    if len(roots)!=len(targets):raise ValueError('Query/target count mismatch')
    return (predictions[roots]-targets).abs().mean()

# %% Task 5: typed receiver aggregation

def typed_messages(x_dict, edge_dict):
    out={kind:torch.zeros_like(x) for kind,x in x_dict.items()}
    for (src,relation,dst),edge in edge_dict.items():
        out[dst].index_add_(0,edge[1],x_dict[src][edge[0]])
    return out

# %% Two-layer heterogeneous temporal regressor (PROVIDED)
class ExamGNN(torch.nn.Module):
    def __init__(self,channels=8):
        super().__init__();self.kinds=['person','event','merchant']
        self.relations=[('event','owner','person'),('event','shop','merchant'),('person','rev_owner','event'),('merchant','rev_shop','event')]
        self.encoders=torch.nn.ModuleDict({k:torch.nn.Linear(2,channels) for k in self.kinds})
        self.root=torch.nn.ModuleList([torch.nn.ModuleDict({k:torch.nn.Linear(channels,channels) for k in self.kinds}) for _ in range(2)])
        self.message=torch.nn.ModuleList([torch.nn.ModuleDict({r:torch.nn.Linear(channels,channels,bias=False) for s,r,d in self.relations}) for _ in range(2)])
        self.head=torch.nn.Linear(channels,1)
    def forward(self,batch):
        x={k:self.encoders[k](batch[k].x) for k in self.kinds}
        for depth in range(2):
            sums={k:torch.zeros_like(v) for k,v in x.items()}
            for src,rel,dst in self.relations:
                transformed={k:v for k,v in x.items()};transformed[src]=self.message[depth][rel](x[src])
                out=typed_messages(transformed,{(src,rel,dst):batch[src,rel,dst].edge_index})
                sums[dst]=sums[dst]+out[dst]
            x={k:torch.relu(self.root[depth][k](x[k])+sums[k]) for k in self.kinds}
        return self.head(x['person']).flatten()

# %% RUN: complete small training and inference pipeline (PROVIDED)
def course_run():
    torch.set_num_threads(1);torch.manual_seed(120)
    # Authored next-period targets. These are not measured customer outcomes.
    train_queries=[(90,4),(10,4),(70,4),(90,7),(10,7),(70,7)]
    targets=torch.tensor([3.,2.,4.,3.,2.,4.]);mature=torch.tensor([6,6,6,9,9,9])
    train=Batch.from_data_list([query_graph(*q) for q in train_queries]);roots=seed_positions(train)
    model=ExamGNN();optimizer=torch.optim.Adam(model.parameters(),lr=.01);trace=[]
    for epoch in range(120):
        model.train();optimizer.zero_grad();pred=model(train)
        loss=seed_loss(pred,roots,targets,mature,10);loss.backward();optimizer.step();trace.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        test=Batch.from_data_list([query_graph(90,10),query_graph(10,10)])
        prediction=model(test)[seed_positions(test)].tolist()
        # Disjoint batching must preserve each query's individual prediction.
        separate=[float(model(query_graph(k,t))[seed_positions(query_graph(k,t))][0]) for k,t in [(90,10),(10,10)]]
    torch.testing.assert_close(torch.tensor(prediction),torch.tensor(separate),atol=1e-5,rtol=1e-5)
    assert trace[-1]<trace[0]*.5, 'Reference training did not learn the course fixture'
    return dict(status='PASS',scope='COURSE_ONLY',initial_mae=trace[0],final_mae=trace[-1],epochs=120,
                test_queries=[[90,10],[10,10]],test_predictions=prediction,batch_vs_single='MATCH',
                split='train query times 4/7 with labels mature by fit10; inference time10',
                learner_status='PENDING_WRITTEN_DEFENSE')
