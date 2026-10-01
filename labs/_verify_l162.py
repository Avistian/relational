"""Independent matrix-walk oracle, explicit pair counting and mutation checks."""
import hashlib,json,signal
from pathlib import Path
from itertools import product
from _check_l162 import check162
from _run_l162 import audit162
from relkit.vision_l162 import reachable_rows,token_budget,evidence_verdict
signal.alarm(600)
p=Path(__file__).resolve().parent;e=p/'evidence/l162'
packet=json.loads((e/'fixtures.json').read_text())
assert check162(reachable_rows,token_budget,evidence_verdict)=='PASS'
# Enumerate all 64 undirected four-node graphs. Integer matrix walks include
# diagonal ones, so walks of h steps include paths of every length <= h.
names=['a','b','c','d'];pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
checked=0
for mask,root,hops,filter_tables in product(range(64),names,range(4),[False,True]):
    nodes={'a':'x','b':'y','c':'x','d':'z'}
    edges=[[names[i],names[j]] for bit,(i,j) in enumerate(pairs) if mask&(1<<bit)]
    allowed=[nodes[root]] if filter_tables else None
    matrix=[[int(i==j or [names[min(i,j)],names[max(i,j)]] in edges) if allowed is None or nodes[names[i]] in allowed and nodes[names[j]] in allowed else 0 for j in range(4)] for i in range(4)]
    state=[int(n==root) for n in names]
    for _ in range(hops):state=[sum(matrix[i][j]*state[j] for j in range(4)) for i in range(4)]
    expected=[n for n,v in zip(names,state) if v]
    assert reachable_rows(nodes,edges,root,hops,allowed)==expected
    checked+=1
pair_cases=0
for lengths in product(range(5),repeat=3):
    for limit in [1,3,8]:
        r=token_budget(list(lengths),limit)
        tokens=[(row,t) for row,n in enumerate(lengths) for t in range(n)]
        assert r['whole_table_pairs']==len([(a,b) for a in tokens for b in tokens])
        assert r['row_pairs']==len([(a,b) for a in tokens for b in tokens if a[0]==b[0]])
        assert r['retained_row_pairs']==len([(a,b) for a in tokens for b in tokens if a[0]==b[0] and a[1]<limit and b[1]<limit])
        pair_cases+=1
for case in packet['evidence_cases']:
    r=evidence_verdict(case)
    assert len(r['missing'])==sum(not x for x in case.values())
    assert (r['supported']=='TRANSFER_REVIEW_ELIGIBLE')==all(case.values())
    assert r['transfer']=='NOT_ESTABLISHED'
mutants=[(lambda *a,**kw:['m1'],token_budget,evidence_verdict),(reachable_rows,lambda *a,**kw:{},evidence_verdict),(reachable_rows,token_budget,lambda *a,**kw:dict(supported='TRANSFER_REVIEW_ELIGIBLE',transfer='PASS',missing=[]))]
for funcs in mutants:
    try:check162(*funcs)
    except (AssertionError,KeyError,ValueError):pass
    else:raise AssertionError('Incorrect learner code escaped')
assert audit162(packet,reachable_rows,token_budget,evidence_verdict)==json.loads((e/'report.json').read_text())
ledger=json.loads((p/'sources/l162/source-ledger.json').read_text())
assert hashlib.sha256((p/'sources/l162/paper.html').read_bytes()).hexdigest()==ledger['sha256']
r=dict(status='PASS',matrix_walk_cases=checked,pair_count_cases=pair_cases,evidence_cases=64,mutants_rejected=3,complete_audit=dict(graph=16,token=8,evidence=64),cloud_spend_usd=0,historical_reproduction='NOT_RUN',transfer='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',sha256={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'relkit/vision_l162.py',p/'_check_l162.py',p/'_run_l162.py',e/'fixtures.json',e/'report.json']})
(p/'_verify_l162_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
