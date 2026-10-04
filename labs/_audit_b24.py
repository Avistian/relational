"""Independent full replay. Authentication precedes semantics; no model is fitted."""
import hashlib, io, json, tempfile, zipfile
from pathlib import Path
import numpy as np
from relkit.defense_b24 import paired_effect, defense_gate, falsification_contract
ARMS=['RDBPFN','RDBPFN_single','TabICLv1.1','Logistic']

def unpack(packet, identity, target):
    raw=Path(packet).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identity['packet_sha256']:raise ValueError('Frozen B23 archive changed')
    target=Path(target).resolve()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            dest=(target/name).resolve()
            if not dest.is_relative_to(target):raise ValueError('Unsafe archive path')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    root=target/'evidence/b23';m=json.loads((root/'packet-manifest.json').read_text())
    for name,digest in m['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed evidence '+name)
    return root

def rank_auc(labels, probabilities):
    """Average ranks for ties, implemented independently of B23's pair-count metric."""
    y=np.asarray(labels);p=np.asarray(probabilities)
    if y.ndim!=1 or p.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid labels/probabilities')
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('AUROC needs both classes')
    order=np.argsort(p,kind='stable');ranks=np.empty(len(p));i=0
    while i<len(p):
        j=i+1
        while j<len(p) and p[order[j]]==p[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def scan(root):
    """Check semantics even if a corrupted receipt has been rehashed by a writer."""
    root=Path(root);records=[];errors=[];baseline_errors=[]
    with np.load(root/'packet/prepared.npz',allow_pickle=False) as d:
        keys=d['test_keys'];labels=d['y_test'];supports=d['support']
        if keys.shape!=(702,2) or len(set(map(tuple,keys)))!=702 or supports.shape!=(10,512):raise ValueError('Wrong complete query/support grid')
        seen=set()
        for phase in ['pilot-1','full-1','baseline-1']:
            receipt=json.loads((root/phase/'receipt.json').read_text())
            for row in receipt['records']:
                arm=row['arm'];s=row['seed']
                if arm not in ARMS or type(s) is not int or s not in range(10) or (arm,s) in seen:raise ValueError('Wrong or duplicate run')
                if row['rows']!=702 or row['support']!=512:raise ValueError('Wrong row count')
                seen.add((arm,s));path=root/phase/f'{arm}-{s}.npz'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Prediction digest mismatch')
                with np.load(path,allow_pickle=False) as x:
                    if not np.array_equal(x['keys'],keys) or not np.array_equal(x['label'],labels) or not np.array_equal(x['support_keys'],d['train_keys'][supports[s]]):raise ValueError('Wrong keys, labels or support identities')
                    if len(set(map(tuple,x['support_keys'])))!=512 or set(map(tuple,keys))&set(map(tuple,x['support_keys'])):raise ValueError('Support leakage/duplicates')
                    value=rank_auc(labels,x['probability']);err=abs(value-row['auc'])
                    if not np.isfinite(row['auc']) or err>1e-12:raise ValueError('Metric receipt mismatch')
                    errors.append(err);records.append(dict(arm=arm,seed=s,auc=value))
                    if arm=='Logistic':
                        X=d['X_train'][supports[s]].copy();Q=d['X_test'].copy()
                        med=np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else 0. for c in X.T])
                        X=np.where(np.isnan(X),med,X).astype(X.dtype).astype(float);Q=np.where(np.isnan(Q),med,Q).astype(Q.dtype)
                        mean=X.mean(0);scale=X.std(0);scale[scale==0]=1
                        if not np.allclose(med,x['median'],rtol=0,atol=1e-12) or not np.allclose(mean,x['mean'],rtol=0,atol=1e-12) or not np.allclose(scale,x['scale'],rtol=1e-10,atol=1e-10):raise ValueError('Baseline fit-state mismatch')
                        # B23 sklearn1.9 float32 transform casts statistics to the input dtype.
                        Q-=mean.astype(Q.dtype);Q/=scale.astype(Q.dtype)
                        prob=1/(1+np.exp(-(Q@x['coef'][0]+x['intercept'][0])))
                        error=float(np.max(np.abs(prob-x['probability'])));baseline_errors.append(error)
                        if error>1e-10:raise ValueError('Baseline probability mismatch')
        if seen!={(a,s) for a in ARMS for s in range(10)}:raise ValueError('Incomplete declared 40-run experiment')
    return records,dict(independent_auc_runs=len(errors),baseline_vectors_reconstructed=len(baseline_errors),max_auc_error=max(errors),max_baseline_error=max(baseline_errors))

def replay(packet, identity, pair=paired_effect, gate=defense_gate, falsify=falsification_contract):
    with tempfile.TemporaryDirectory(prefix='b24-audit-') as td:
        root=unpack(packet,identity,td);records,checks=scan(root)
        old=json.loads((root/'report.json').read_text());models={}
        for arm in ARMS:
            v=[r['auc'] for r in sorted(records,key=lambda r:r['seed']) if r['arm']==arm]
            if not np.allclose(v,old['models'][arm]['per_seed'],rtol=0,atol=1e-12):raise ValueError('B23 per-draw parity failed')
            models[arm]=dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),per_seed=v,lane='COURSE_BASELINE' if arm=='Logistic' else 'PUBLISHED_SELECTED')
        comparisons=[pair(records,a,'RDBPFN') for a in ARMS[1:]]
        # These deliberately hypothetical cases exercise gates, not learner scoring.
        scores=dict.fromkeys(['protocol','baselines','reproducibility','interpretation','falsifiability'],2)
        from _test_b24 import fixtures
        _,example_tests=fixtures()
        return dict(experiment='B24-RESEARCH-DEFENSE-AUDIT',execution='COMPLETE_SAVED_EVIDENCE_REPLAY',runs=40,predictions=28080,paper_runs=30,course_runs=10,models=models,comparisons=comparisons,verification=checks,source_packet_sha256=identity['packet_sha256'],historical_identity='NOT_ESTABLISHED',historical_feature_availability='NOT_ESTABLISHED',fresh_inference='NOT_RUN',fresh_pretraining='NOT_RUN',full_dfs_regeneration='NOT_RUN',whole_paper='NOT_RUN',untouched_task='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',illustrative_gate_missing_reproduction=gate(scores,False,False,False),illustrative_gate_unresolved_leakage=gate(scores,True,True,False),illustrative_falsification_contract=falsify(example_tests,['rel-f1/driver-dnf']))
if __name__=='__main__':
    p=Path(__file__).resolve().parent;e=p/'evidence/b24';identity=json.loads((e/'source-identity.json').read_text())
    r=replay(p/'evidence/b23/portable-packet.zip',identity)
    (e/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['execution'],r['verification'])
