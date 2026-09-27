"""Independent mechanism oracles, rejected faults, source and dispatch safeguards."""
import ast,copy,hashlib,itertools,json,tempfile
from pathlib import Path
import numpy as np
from _check_l119 import check_ceiling,check_typed,check_time,check_mae
from relkit.synthesis_l119 import collision_ceiling,typed_sum,eligible_nodes,aligned_mae,course_experiment
P=Path(__file__).resolve().parent;R=P.parent
for check,fn in [(check_ceiling,collision_ceiling),(check_typed,typed_sum),(check_time,eligible_nodes),(check_mae,aligned_mae)]:check(fn)
rng=np.random.default_rng(119)
for _ in range(40):
    groups=rng.integers(0,3,9);y=rng.integers(0,2,9)
    # Exhaust all eight deterministic labelings of three observed feature groups.
    oracle=max(sum(rule[g]==label for g,label in zip(groups,y))/len(y) for rule in itertools.product([0,1],repeat=3))
    assert collision_ceiling([(int(g),) for g in groups],y)==oracle
    x=rng.normal(size=5);edges=[(int(rng.integers(5)),int(rng.integers(5)),str(rng.integers(2))) for _ in range(12)]
    matrices={r:np.zeros((5,5)) for r in ['0','1']};weights={'0':.7,'1':-1.3}
    for src,dst,rel in edges:matrices[rel][dst,src]+=1
    np.testing.assert_allclose(typed_sum(x,edges,weights),sum(weights[k]*(a@x) for k,a in matrices.items()),atol=1e-12)
    ids=[f'query-{i}' for i in range(8)];pred=rng.normal(size=8);target=rng.normal(size=8);order=rng.permutation(8)
    assert abs(aligned_mae([ids[i] for i in order],pred[order],ids,target)-float(np.mean(abs(pred-target))))<1e-12

def unweighted_groups(features,labels):return 1.
def erase_types(x,edges,weights):return typed_sum(x,edges,{k:1. for k in weights})
def event_only(edges,event,available,seed,cutoff,hops):return eligible_nodes(edges,event,event,seed,cutoff,hops)
def no_cutoff(edges,event,available,seed,cutoff,hops):return eligible_nodes(edges,event,available,seed,float('inf'),hops)
def positional_mae(pi,p,ti,t):return float(np.mean(np.abs(np.asarray(p)-t)))
faults=[]
for check,fn in [(check_ceiling,unweighted_groups),(check_typed,erase_types),(check_time,event_only),(check_time,no_cutoff),(check_mae,positional_mae)]:
    try:check(fn)
    except (AssertionError,ValueError):faults.append(fn.__name__)
assert len(faults)==5
course=course_experiment();assert course['indistinguishable'] and course['coarse_ceiling']==.5 and course['augmented_ceiling']==1.

budget=json.loads((P/'_budget_l119.json').read_text())
for name,digest in budget['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
assert budget['maximum_worker_usd']+budget['overhead_reserve_usd']<=budget['budget_usd']
assert sum(r['workers'] for r in budget['reservations'])==6
# Exercise pre-dispatch rejection without a Modal connection or a worker call.
tree=ast.parse((R/'modal/l119_repro.py').read_text());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');entry.decorator_list=[]
class NeverLaunch:
    def remote(self,*a):raise RuntimeError('A guard failed: remote dispatch reached')
    def starmap(self,*a):raise RuntimeError('A guard failed: remote dispatch reached')
guards=[]
with tempfile.TemporaryDirectory(prefix='l119-budget-') as tmp:
    root=Path(tmp);local=root/'ledger.json';(root/'source.py').write_text('frozen source')
    base=copy.deepcopy(budget);base['source_hashes']={'source.py':hashlib.sha256((root/'source.py').read_bytes()).hexdigest()}
    scope={'ROOT':root,'RATE':.00022572,'worker':NeverLaunch()}
    exec(compile(ast.Module(body=[entry],type_ignores=[]),'<budget-entry>','exec'),scope)
    for label,change in [('duplicate',{}),('changed-source',{'source_hashes':{'source.py':'bad'}}),('unapproved',{'reservations':[],'pilot_approved_for_full':False}),('exhausted',{'reservations':[{'mode':'other','workers':8}]})]:
        b=copy.deepcopy(base);b.update(change);local.write_text(json.dumps(b))
        try:scope['main']('paper','ledger.json')
        except AssertionError:guards.append(label)
        else:raise AssertionError('Guard did not reject '+label)
assert len(guards)==4
uuids=[]
for seed in range(5):
    root=P/f'evidence/l119/paper/seed-{seed}'
    done=json.loads((root/'completed.json').read_text());uuids.append(done['run_uuid'])
    assert done['lesson']==119 and done['epochs']==10 and done['status']=='COMPLETE'
    assert done['started_utc']>=budget['reservations'][1]['utc'] and done['finished_utc']>done['started_utc']
assert len(set(uuids))==5
report=dict(status='PASS',independent_random_cases=120,rejected_mutations=faults,dispatch_guards=guards,
            course=course,fresh_unique_runs=5,source_files_unchanged=len(budget['source_hashes']))
(P/'_verify_l119_results.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='course'})
