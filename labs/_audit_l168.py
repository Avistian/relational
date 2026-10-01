"""Authenticate and independently rescore both selected databases."""
def audit168(root,pins,auc_fn,paired_fn,macro_fn):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    root=Path(root)
    for name,digest in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch: '+name)
    all_records=[];datasets={};raw_count=0
    for spec in pins['experiments']:
        folder=root/spec['folder'];manifest=json.loads((folder/'input-manifest.json').read_text())
        if pins['files'][spec['folder']+'/prepared.npz']!=manifest['files']['prepared.npz']['sha256']:raise ValueError('Original data hash differs')
        data=np.load(folder/'prepared.npz',allow_pickle=False)
        keys=data['test_keys'];labels=data['y_test'];train=data['train_keys'];supports=data['support']
        if len(keys)!=spec['test_rows'] or len(train)!=spec['train_rows'] or data['X_test'].shape!=(len(keys),spec['features']):raise ValueError('Changed task shape')
        if len(set(map(tuple,keys)))!=len(keys) or len(set(map(tuple,train)))!=len(train):raise ValueError('Duplicate query identity')
        if set(map(tuple,train))&set(map(tuple,keys)):raise ValueError('Train/test key overlap')
        if supports.shape!=(10,512):raise ValueError('Incomplete context schedule')
        for seed in range(10):
            integer=int.from_bytes(hashlib.sha256(spec['seed_key'].format(seed=seed).encode()).digest()[:4],'big')
            expected=np.random.default_rng(integer).choice(len(train),512,replace=False)
            if not np.array_equal(expected,supports[seed]):raise ValueError('Different support schedule')
        expected={(a,s) for a in spec['targets'] for s in range(10)};seen=set();scores={a:{} for a in spec['targets']}
        for phase in spec['phases']:
            receipt=json.loads((folder/phase/'receipt.json').read_text())
            if receipt['input_manifest_sha256']!=pins['files'][spec['folder']+'/input-manifest.json']:raise ValueError('Original receipt input differs')
            for r in receipt['records']:
                arm,seed=r['arm'],r['seed'];identity=(arm,seed)
                if identity not in expected or identity in seen:raise ValueError('Duplicate or unexpected run')
                seen.add(identity);path=folder/phase/f'{arm}-{seed}.npz'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Original receipt hash differs')
                with np.load(path,allow_pickle=False) as out:
                    if not np.array_equal(out['support_keys'],train[supports[seed]]):raise ValueError('Different support identities')
                    if len(out['keys'])!=len(keys) or len(set(map(tuple,out['keys'])))!=len(keys):raise ValueError('Incomplete prediction keys')
                    if dict(zip(map(tuple,out['keys']),out['label']))!=dict(zip(map(tuple,keys),labels)):raise ValueError('Labels differ by query key')
                    auc=auc_fn(keys,labels,out['keys'],out['probability'])
                    if abs(auc-r['auc'])>1e-12:raise ValueError('Metric receipt differs')
                raw_count+=len(keys);scores[arm][seed]=auc
                if arm!='RDBPFN_single':all_records.append(dict(database=spec['database'],seed=seed,arm=arm,auc=auc,test_rows=len(keys),metric='AUROC'))
        if seen!=expected:raise ValueError('Missing run')
        models={}
        for arm,target in spec['targets'].items():
            v=np.array([scores[arm][s] for s in range(10)])
            models[arm]=dict(per_seed=v.tolist(),mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),paper=target,delta=float(v.mean()-target),
                            status='CLOSE' if abs(v.mean()-target)<=.02 else 'OUTSIDE_TOLERANCE')
        datasets[spec['database']]=dict(task=spec['task'],evidence=spec['evidence'],test_rows=len(keys),features=spec['features'],models=models)
    paired=paired_fn(all_records,[s['database'] for s in pins['experiments']],list(range(10)))
    return dict(experiment='L168 Table9 rel-trial/study-outcome fresh + rel-f1/driver-dnf reused',status='COMPLETE_SELECTED_RELEASED_CHECKPOINT_EVALUATION',
                fresh_runs=30,reused_runs=30,fresh_predictions=24750,reused_predictions=21060,all_predictions_checked=raw_count,
                datasets=datasets,paired=paired,macro=macro_fn(paired),hashed_files=len(pins['files']),
                historical_identity='NOT_ESTABLISHED',checkpoint_training_lineage='NOT_ESTABLISHED',exact_target_schema_exclusion='NOT_ESTABLISHED',
                fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',griffin_fresh='INCOMPLETE_BUDGET_GATE',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import json,signal
    from pathlib import Path
    from relkit.transfer_l167 import keyed_auc
    from relkit.generalization_l168 import paired_gains,database_macro
    signal.alarm(600);p=Path(__file__).resolve().parent
    result=audit168(p,json.loads((p/'evidence/l168/audit-manifest.json').read_text()),keyed_auc,paired_gains,database_macro)
    (p/'evidence/l168/report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
