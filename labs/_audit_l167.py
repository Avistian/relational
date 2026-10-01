"""Independent audit of L166 raw predictions; never dispatches inference."""
def audit167(root, pins, auc_fn):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    root=Path(root)
    for name,digest in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Artifact hash mismatch: '+name)
    e=root/'evidence/l166'
    manifest=json.loads((e/'input-manifest.json').read_text())
    if pins['files']['evidence/l166/prepared.npz']!=manifest['files']['prepared.npz']['sha256']:
        raise ValueError('Original prepared-data hash mismatch')
    source=json.loads((root/'sources/l166/source-ledger.json').read_text())
    for name,digest in pins['files'].items():
        if name.startswith('sources/l166/upstream/'):
            if digest!=source['files'][name.removeprefix('sources/l166/')]:
                raise ValueError('Original source hash mismatch')
    with np.load(e/'prepared.npz',allow_pickle=False) as data:
        train=data['train_keys'];keys=data['test_keys'];labels=data['y_test'];supports=data['support']
        if train.shape!=(11411,2) or keys.shape!=(702,2) or data['X_train'].shape!=(11411,72) or data['X_test'].shape!=(702,72):
            raise ValueError('Selected population changed')
        if len(set(map(tuple,train)))!=len(train) or len(set(map(tuple,keys)))!=len(keys):
            raise ValueError('Duplicate prepared identities')
        if set(map(tuple,train))&set(map(tuple,keys)):raise ValueError('Train/test key overlap')
        if supports.shape!=(10,512):raise ValueError('Incomplete support schedule')
        for seed in range(10):
            integer=int.from_bytes(hashlib.sha256(f'rel-f1-dfs-2:driver-dnf:{seed}'.encode()).digest()[:4],'big')
            indices=np.random.default_rng(integer).choice(len(train),512,replace=False)
            if not np.array_equal(indices,supports[seed]):raise ValueError('Support seed schedule mismatch')
    targets={'RDBPFN':.7219,'RDBPFN_single':.6640,'TabICLv1.1':.7176}
    expected={(a,s) for a in targets for s in range(10)};seen=set();scores={a:{} for a in targets}
    manifest_hash=hashlib.sha256((e/'input-manifest.json').read_bytes()).hexdigest()
    for folder in ['pilot-2','full-1']:
        receipt=json.loads((e/folder/'receipt.json').read_text())
        if receipt['input_manifest_sha256']!=manifest_hash:raise ValueError('Original receipt input hash mismatch')
        for row in receipt['records']:
            arm,seed=row['arm'],row['seed'];identity=(arm,seed)
            if identity not in expected or identity in seen:raise ValueError('Unexpected/duplicate selected run')
            seen.add(identity);path=e/folder/f'{arm}-{seed}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Original receipt prediction hash mismatch')
            with np.load(path,allow_pickle=False) as result:
                if not np.array_equal(result['support_keys'],train[supports[seed]]):raise ValueError('Unpaired support identities')
                # Check labels by key independently of serialization order.
                if len(result['keys'])!=len(keys) or len(set(map(tuple,result['keys'])))!=len(keys):raise ValueError('Incomplete query keys')
                label_map=dict(zip(map(tuple,result['keys']),result['label']))
                if label_map!=dict(zip(map(tuple,keys),labels)):raise ValueError('Labels differ by query key')
                score=auc_fn(keys,labels,result['keys'],result['probability'])
                if abs(score-row['auc'])>1e-12:raise ValueError('Original receipt metric mismatch')
                scores[arm][seed]=score
    if seen!=expected:raise ValueError('Missing selected runs')
    models={};vectors={}
    for arm,target in targets.items():
        v=np.array([scores[arm][seed] for seed in range(10)]);vectors[arm]=v
        models[arm]=dict(per_seed=v.tolist(),mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),paper=target,
                         status='CLOSE' if abs(v.mean()-target)<=.02 else 'OUTSIDE_TOLERANCE')
    paired={}
    for arm in ['RDBPFN_single','TabICLv1.1']:
        d=vectors['RDBPFN']-vectors[arm]
        paired[arm]=dict(per_seed=d.tolist(),mean=float(d.mean()),sample_sd=float(d.std(ddof=1)),
                         min=float(d.min()),max=float(d.max()),positive_seeds=int((d>0).sum()))
    return dict(experiment=pins['scope'],status='COMPLETE_SAVED_EVIDENCE_AUDIT',runs=30,predictions=21060,
                queries=702,features=72,support_rows=512,seeds=list(range(10)),models=models,paired=paired,
                hashed_files=len(pins['files']),support_schedule='INDEPENDENTLY_REGENERATED',
                new_benchmark_inference='NOT_RUN',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',
                historical_identity='NOT_ESTABLISHED',additional_paid_compute_usd=0,learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import json,signal
    from pathlib import Path
    from relkit.transfer_l167 import keyed_auc
    signal.alarm(600)
    p=Path(__file__).resolve().parent
    result=audit167(p,json.loads((p/'evidence/l167/input-manifest.json').read_text()),keyed_auc)
    (p/'evidence/l167/report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
