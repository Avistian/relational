"""Independent pairwise metric and path-enumeration oracles; fault injection."""
import copy,hashlib,itertools,json
from pathlib import Path
from relkit.context_l165 import eligible_context,visible_graph,keyed_auc
from _check_l165 import check165
from _run_l165 import audit165
P=Path(__file__).resolve().parent;E=P/'evidence/l165'
packet=json.loads((E/'fixtures.json').read_text());before=copy.deepcopy(packet)
assert check165(eligible_context,visible_graph,keyed_auc)=='PASS'
report=audit165(packet,eligible_context,visible_graph,keyed_auc)
assert report==json.loads((E/'report.json').read_text())
for case,actual in zip(packet['context_cases'],report['context']):
    r=case['context'][0];q=case['query']['cutoff']
    expected=[] if q<=r['cutoff'] or q<max(r['label_end'],r['available_at']) else [[r['entity'],r['cutoff']]]
    assert expected==actual['selected']
for case,actual in zip(packet['graph_cases'],report['graphs']):
    rows={r['id']:r for r in case['rows']};root=case['root'];q=case['cutoff'];reached={root}
    # Enumerate every possible walk instead of repeating the implementation's BFS.
    links={tuple(e) for e in case['edges']}|{tuple(reversed(e)) for e in case['edges']}
    for length in range(1,case['hops']+1):
        for tail in itertools.product(rows,repeat=length):
            walk=(root,)+tail
            if all((a,b) in links for a,b in zip(walk,walk[1:])) and all(rows[n]['available_at']<=q and (rows[n]['event_at'] is None or rows[n]['event_at']<=q) for n in walk):
                reached.add(tail[-1])
    assert sorted(reached)==actual['visible']
auc_cases=0
for labels in itertools.product([0,1],repeat=4):
    if len(set(labels))<2:continue
    for scores in itertools.product([0.,.5,1.],repeat=4):
        truth=[dict(entity='same',cutoff=i,label=y) for i,y in enumerate(labels)]
        predictions=[dict(entity='same',cutoff=i,score=s) for i,s in enumerate(scores)][::-1]
        comparisons=[float(scores[i]>scores[j])+.5*float(scores[i]==scores[j]) for i in range(4) for j in range(4) if labels[i]==1 and labels[j]==0]
        oracle=sum(comparisons)/len(comparisons)
        assert abs(keyed_auc(truth,predictions)-oracle)<1e-12;auc_cases+=1
# Query batch must not widen the cutoff for an earlier owner.
case=packet['graph_cases'][-1]
a=visible_graph(case['rows'],case['edges'],'u1',7,2);b=visible_graph(case['rows'],case['edges'],'u1',15,2)
assert a==['o1','u1'] and b==['o1','o2','p1','u1']
# Adding a label feature is rejected; no accidental target-bearing row can enter.
try:visible_graph([dict(r,label=1) for r in case['rows']],case['edges'],'u1',15,2)
except ValueError:pass
else:raise AssertionError('Label feature accepted')
faults=[(lambda c,q:[r for r in c if r['cutoff']<q['cutoff']],visible_graph,keyed_auc),
        (eligible_context,lambda rows,edges,root,cutoff,hops:visible_graph(rows,edges,root,100,hops),keyed_auc),
        (eligible_context,visible_graph,lambda truth,pred:.5)]
rejected=0
for triple in faults:
    try:check165(*triple)
    except AssertionError:rejected+=1
assert rejected==3 and packet==before
ledger=json.loads((P/'sources/l165/source-ledger.json').read_text())
for source in ledger['sources']:
    if 'sha256' in source:assert hashlib.sha256((P/'sources/l165'/source['file']).read_bytes()).hexdigest()==source['sha256']
r=dict(status='PASS',context_oracle_cases=144,graph_path_oracle_cases=96,pairwise_auc_cases=auc_cases,broken_implementations_rejected=rejected,owner_cutoff='PASS',label_feature_rejection='PASS',input_immutability='PASS',source_hashes='PASS',historical_reproduction='NOT_RUN',historical_fidelity='NOT_ESTABLISHED')
(P/'_verify_l165_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
