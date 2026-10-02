"""Independent schedule, rank, completeness and preserved-source checks."""
import ast,copy,hashlib,json,math,tempfile,shutil
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from relkit.few_shot_l176 import nested_support,keyed_auc,paired_curve,reserve_cost
from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
from _check_l176 import checks
from _audit_l169 import audit169
from _audit_l176 import audit176
P=Path(__file__).resolve().parent;E=P/'evidence/l176'
assert checks(nested_support,keyed_auc,paired_curve,reserve_cost)=='PASS'
wrong=[(lambda n,ks,t:{k:np.arange(k) for k in ks},keyed_auc,paired_curve,reserve_cost),(nested_support,lambda *a:.875,paired_curve,reserve_cost),(nested_support,keyed_auc,lambda r,ks,ss:paired_curve(r[:-1],ks,ss),reserve_cost)]
for funcs in wrong:
    try:checks(*funcs)
    except (AssertionError,ValueError,KeyError):pass
    else:raise AssertionError('Incorrect learner function accepted')
rng=np.random.default_rng(176);error=0
for n in range(2,202):
    y=np.r_[0,1,rng.integers(0,2,n-2)];p=rng.integers(0,10,n)/9;keys=np.column_stack([np.arange(n)//2,np.arange(n)%2]);order=rng.permutation(n)
    score=keyed_auc(keys,y,keys[order],p[order]);error=max(error,abs(score-roc_auc_score(y,p)));assert error<1e-12
for _ in range(100):
    values=rng.uniform(.3,.9,size=(5,10));rs=[dict(context=k,seed=s,auc=float(values[i,s])) for i,k in enumerate([64,128,256,512,1024]) for s in range(10)];rng.shuffle(rs)
    c=paired_curve(rs,[64,128,256,512,1024],list(range(10)))
    np.testing.assert_allclose([x['mean'] for x in c['levels']],values.mean(axis=1),atol=1e-14)
    np.testing.assert_allclose([x['mean_gain'] for x in c['doublings']],np.diff(values,axis=0).mean(axis=1),atol=1e-14)
# Authenticate the full original experiment without mutating inherited reports.
oldpins=json.loads((E/'inherited-manifest.json').read_text())
old=audit169(P,oldpins,keyed_auc,sample_context,scaling_curve,scaling_claim)
assert old==json.loads((P/'evidence/l169/report.json').read_text())
(E/'published-replay.json').write_text(json.dumps(old,indent=2)+'\n')
# The original model calls and prediction pipeline must remain byte-identical.
a=(P/'_run_l169.py').read_text();b=(P/'_run_l176.py').read_text()
assert a.split('    for spec in manifest')[1].split('    expected=')[0]==b.split('    for spec in manifest')[1].split('    expected=')[0]
if not (E/'audit-manifest.json').exists():
    print('Pre-inference contracts PASS; full published replay',old['all_predictions_checked']);raise SystemExit(0)
pins=json.loads((E/'audit-manifest.json').read_text());report=audit176(P,pins,nested_support,keyed_auc,paired_curve)
count=0;max_error=0
for phase in ['pilot-1','remaining-1']:
    receipt=json.loads((E/phase/'receipt.json').read_text())
    for r in receipt['records']:
        d=np.load(E/phase/r['filename']);y=d['label'];p=d['probability'];n1=int(y.sum());n0=len(y)-n1
        rank=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
        max_error=max(max_error,abs(rank-r['auc']),abs(roc_auc_score(y,p)-r['auc']));count+=len(y)
assert max_error<1e-12 and count==229050
for spec in json.loads((E/'input-manifest.json').read_text())['experiments']:
    d=np.load(E/(spec['database']+'.npz'))
    for seed in range(10):
        integer=int.from_bytes(hashlib.sha256(spec['seed_key'].format(seed=seed).encode()).digest()[:4],'big')
        expected=np.random.Generator(np.random.PCG64(integer)).choice(len(d['y_train']),1024,replace=False)
        for k in [64,128,256,512,1024]:np.testing.assert_array_equal(d['support_'+str(k)][seed],expected[:k])
# A mutated packet cannot be made valid by hiding its missing rows.
corruptions=0
with tempfile.TemporaryDirectory(prefix='l176-corruption-') as tmp:
    root=Path(tmp)
    for name in pins['files']:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(P/name)
    rel='evidence/l176/pilot-1/receipt.json';path=root/rel;path.unlink();original=(P/rel).read_text()
    for kind in ['missing','duplicate','metric','manifest']:
        obj=json.loads(original)
        if kind=='missing':obj['records'].pop()
        elif kind=='duplicate':obj['records'].append(obj['records'][0])
        elif kind=='metric':obj['records'][0]['auc']+=.1
        else:obj['input_manifest_sha256']='bad'
        path.write_text(json.dumps(obj));changed=copy.deepcopy(pins);changed['files'][rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        try:audit176(root,changed,nested_support,keyed_auc,paired_curve)
        except ValueError:corruptions+=1
        else:raise AssertionError('Corruption accepted: '+kind)
(E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
r=dict(status='PASS',published_evaluations=300,published_predictions=old['all_predictions_checked'],fresh_evaluations=300,fresh_predictions=count,independent_rank_sklearn_error=max_error,synthetic_metric_cases=200,randomized_pairing_cases=100,wrong_learner_functions_rejected=3,corrupt_receipts_rejected=corruptions,original_predictor_operations='BYTE_IDENTICAL',source_sampling_change='DECLARED_NESTED_PREFIX',historical_identity='NOT_ESTABLISHED')
(P/'_verify_l176_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
