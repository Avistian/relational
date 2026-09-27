"""Independent temporal traversal, rejected learner faults and real PyG checks."""
import json,itertools,hashlib,ast,copy,tempfile
from pathlib import Path
import torch
from _check_l120 import CHECKS
from relkit import exam_l120 as m
P=Path(__file__).resolve().parent;R=P.parent
for name,check in CHECKS:check(getattr(m,name))
# Independent undirected adjacency oracle constructed directly from raw identities.
cases=0
for key,cutoff,hops in itertools.product(m.PERSON_KEYS,range(1,13),range(4)):
    adjacency={}
    for i,(owner,shop,event,arrival) in enumerate(zip(m.EVENT_PERSON,m.EVENT_MERCHANT,m.EVENT_TIME,m.EVENT_AVAILABLE)):
        if max(event,arrival)>cutoff:continue
        for a,b in [(('event',i),('person',m.PERSON_KEYS.index(owner))),(('event',i),('merchant',m.MERCHANT_KEYS.index(shop)))]:
            adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
    seen={('person',m.PERSON_KEYS.index(key))}
    for _ in range(hops):seen|={v for u in list(seen) for v in adjacency.get(u,[])}
    graph=m.query_graph(key,cutoff,hops)
    got={(kind,int(n)) for kind in graph.node_types for n in graph[kind].n_id}
    assert got==seen,(key,cutoff,hops,got,seen)
    for src,rel,dst in graph.edge_types:
        for a,b in graph[src,rel,dst].edge_index.t().tolist():
            assert (dst,int(graph[dst].n_id[b])) in adjacency[(src,int(graph[src].n_id[a]))]
    cases+=1
mutants={
 'key_edges':lambda pk,fk:torch.tensor([[0,2],[10,90]]),
 'visible_rows':lambda e,a,c:torch.tensor(e)<=c,
 'seed_positions':lambda b:torch.arange(b.num_graphs),
 'seed_loss':lambda p,r,y,a,c:(p[r]-y).abs().mean(),
 'typed_messages':lambda x,e:{k:torch.zeros_like(v) for k,v in x.items()}}
rejected=[]
for name,check in CHECKS:
    try:check(mutants[name])
    except (AssertionError,ValueError):rejected.append(name)
assert len(rejected)==5
report=m.course_run();report.update(temporal_oracle_cases=cases,rejected_mutants=rejected)
# Source fingerprint and bounded dispatch, with no cloud calls during negative checks.
budget=json.loads((P/'_budget_l120.json').read_text())
for path,digest in budget['source_hashes'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
entry=next(n for n in ast.parse((R/'modal/l120_repro.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');entry.decorator_list=[]
class NeverLaunch:
    def remote(self,*a):raise RuntimeError('Guard failed')
    def starmap(self,*a):raise RuntimeError('Guard failed')
guards=[]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'source').write_text('frozen');base=copy.deepcopy(budget);base['source_hashes']={'source':hashlib.sha256(b'frozen').hexdigest()}
    scope={'ROOT':root,'RATE':.00022572,'worker':NeverLaunch()};exec(compile(ast.Module(body=[entry],type_ignores=[]),'<guard>','exec'),scope)
    for label,change in [('duplicate',{}),('changed-source',{'source_hashes':{'source':'bad'}}),('pilot-required',{'reservations':[],'pilot_approved_for_full':False}),('budget-exhausted',{'reservations':[{'mode':'other','workers':8}]})]:
        b=copy.deepcopy(base);b.update(change);(root/'ledger.json').write_text(json.dumps(b))
        try:scope['main']('paper','ledger.json')
        except AssertionError:guards.append(label)
        else:raise AssertionError(label)
report['dispatch_guards']=guards
(P/'_verify_l120_results.json').write_text(json.dumps(report,indent=2));print(report)
