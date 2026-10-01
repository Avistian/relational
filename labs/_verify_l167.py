"""Independent oracles, adversarial learner checks, and complete evidence audit."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,signal,tempfile,shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from relkit.transfer_l167 import temporal_summary,eligible_support,keyed_auc
from _check_l167 import check167
from _audit_l167 import audit167
signal.alarm(600)
P=Path(__file__).resolve().parent
assert check167(temporal_summary,eligible_support,keyed_auc)=='PASS'
rng=np.random.default_rng(167)
for trial in range(100):
    y=np.array([0,1]+rng.integers(0,2,18).tolist());scores=rng.integers(0,6,20)/5
    keys=np.array([[i%5,i//5] for i in range(20)]);order=rng.permutation(20)
    v=keyed_auc(keys,y,keys[order],scores[order])
    assert abs(v-roc_auc_score(y,scores))<1e-12
    pairs=[float(a>b)+.5*float(a==b) for a in scores[y==1] for b in scores[y==0]]
    assert abs(v-np.mean(pairs))<1e-12
    events=np.column_stack([rng.integers(0,4,20),rng.integers(0,30,20),rng.normal(size=20)])
    queries=[[i,t] for i in range(4) for t in [0,10,20,30]]
    expected=[]
    for entity,time in queries:
        selected=[v for key,t,v in events if key==entity and t<time]
        expected.append([len(selected),sum(selected)/len(selected) if selected else 0])
    np.testing.assert_allclose(temporal_summary(queries,events),expected,atol=1e-12)
    keys=np.array([[i,2*i] for i in range(10)]);ready=keys[:,1]+rng.integers(0,15,10)
    for t in [0,7,15,30]:
        expected=[i for i,((_,q),r) in enumerate(zip(keys,ready)) if q<t and r<t]
        np.testing.assert_array_equal(eligible_support(keys,ready,t),expected)
wrong=[(lambda q,e:np.zeros((len(q),2)),eligible_support,keyed_auc),
       (temporal_summary,lambda k,r,t:np.flatnonzero(np.array(k)[:,1]<t),keyed_auc),
       (temporal_summary,eligible_support,lambda k,y,g,p:roc_auc_score(y,p))]
for functions in wrong:
    try:check167(*functions)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Incorrect learner solution passed')
pins=json.loads((P/'evidence/l167/input-manifest.json').read_text())
result=audit167(P,pins,keyed_auc)
assert result['runs']==30 and result['predictions']==21060
for arm,values in result['models'].items():
    old=json.loads((P/'evidence/l166/report.json').read_text())['models'][arm]
    np.testing.assert_allclose(values['per_seed'],old['per_seed'],atol=1e-12,rtol=0)
with tempfile.TemporaryDirectory(prefix='l167-corrupt-') as tmp:
    root=Path(tmp)
    for name in pins['files']:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,dest)
    target=root/'evidence/l166/full-1/RDBPFN-1.npz';raw=target.read_bytes();target.write_bytes(raw+b'corrupted')
    try:audit167(root,pins,keyed_auc)
    except ValueError as e:assert 'hash' in str(e).lower()
    else:raise AssertionError('Corrupted prediction accepted')
    target.write_bytes(raw)
    # Re-pin the deliberately altered receipt so the semantic completeness check,
    # rather than the outer hash check, must reject missing and duplicate runs.
    name='evidence/l166/full-1/receipt.json';path=root/name;original=path.read_bytes()
    for mode in ['missing','duplicate']:
        receipt=json.loads(original)
        if mode=='missing':receipt['records'].pop()
        else:receipt['records'].append(receipt['records'][0])
        path.write_text(json.dumps(receipt));changed=json.loads(json.dumps(pins))
        changed['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        try:audit167(root,changed,keyed_auc)
        except ValueError as e:assert ('Missing' if mode=='missing' else 'duplicate') in str(e)
        else:raise AssertionError('Invalid run population accepted')
    path.write_bytes(original)
    # Corrupt label semantics while maintaining valid hashes and complete keys.
    name='evidence/l166/full-1/RDBPFN-1.npz';path=root/name
    with np.load(path) as a:arrays={k:a[k] for k in a.files}
    arrays['label']=1-arrays['label'];np.savez_compressed(path,**arrays)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();changed=json.loads(json.dumps(pins));changed['files'][name]=digest
    receipt=json.loads(original)
    for row in receipt['records']:
        if row['arm']=='RDBPFN' and row['seed']==1:row['sha256']=digest
    receipt_path=root/'evidence/l166/full-1/receipt.json';receipt_path.write_text(json.dumps(receipt))
    changed['files']['evidence/l166/full-1/receipt.json']=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    try:audit167(root,changed,keyed_auc)
    except ValueError as e:assert 'Labels differ' in str(e)
    else:raise AssertionError('Reoriented labels accepted')
report=dict(status='PASS',auc_oracle_cases=100,temporal_query_cases=1600,support_cutoff_cases=400,
            incorrect_learner_functions_rejected=3,corrupted_artifact_rejected=True,semantic_corruptions_rejected=3,raw_runs=30,
            original_metric_agreement='WITHIN_1e-12',additional_paid_compute_usd=0)
(P/'_verify_l167_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
