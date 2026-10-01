"""Authenticate and independently rescore every selected context, model and seed."""
def audit169(root,pins,auc_fn,sample_fn,curve_fn,claim_fn):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    root=Path(root)
    for name,digest in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch: '+name)
    manifest=json.loads((root/'evidence/l169/input-manifest.json').read_text())
    original=json.loads((root/'sources/l166/source-ledger.json').read_text())
    for name,digest in manifest['source_files'].items():
        if digest!=original['files'][name.removeprefix('sources/l166/')]:raise ValueError('Original source ledger differs')
        if pins['files'][name]!=digest:raise ValueError('Source pin differs')
    for db in manifest['experiments']:
        name=db['database']+'.npz'
        if pins['files']['evidence/l169/'+name]!=manifest['files'][name]['sha256']:raise ValueError('Input data pin differs')
    rows=[];raw_count=0;fresh_count=0;reused_count=0;seen=set();curves={};pairs={};comparisons=[];support_overlaps={}
    for spec in manifest['experiments']:
        db=spec['database'];data=np.load(root/'evidence/l169'/(db+'.npz'),allow_pickle=False)
        keys=data['test_keys'];labels=data['y_test'];train=data['train_keys'];oldfolder=root/spec['original_folder'];old=json.loads((oldfolder/'input-manifest.json').read_text())
        olddata=np.load(oldfolder/'prepared.npz',allow_pickle=False)
        if hashlib.sha256((oldfolder/'prepared.npz').read_bytes()).hexdigest()!=old['files']['prepared.npz']['sha256']:raise ValueError('Original data hash mismatch')
        for key in ['X_train','X_test','y_train','y_test','train_keys','test_keys']:
            if not np.array_equal(data[key],olddata[key],equal_nan=True):raise ValueError('Changed original task array: '+key)
        if len(keys)!=spec['test_rows'] or len(train)!=spec['train_rows'] or data['X_test'].shape!=(len(keys),spec['features']):raise ValueError('Wrong task shape')
        if len(set(map(tuple,keys)))!=len(keys) or len(set(map(tuple,train)))!=len(train) or set(map(tuple,keys))&set(map(tuple,train)):raise ValueError('Invalid query keys')
        horizon=(60 if db=='rel-f1' else 365)*86400*10**9
        if not (train[:,1]+horizon<keys[:,1].min()).all():raise ValueError('Support labels not ready')
        for k in manifest['contexts']:
            schedule=data['support_'+str(k)]
            if schedule.shape!=(10,k):raise ValueError('Incomplete support schedule')
            for seed in range(10):
                expected=sample_fn(len(train),k,spec['seed_key'].format(seed=seed))
                if not np.array_equal(schedule[seed],expected):raise ValueError('Different support schedule')
            if k==512 and not np.array_equal(schedule,olddata['support']):raise ValueError('Reused supports differ')
        support_overlaps[db]=[dict(start=a,end=b,overlap_counts=[len(set(data['support_'+str(a)][s])&set(data['support_'+str(b)][s])) for s in range(10)]) for a,b in zip(manifest['contexts'][:-1],manifest['contexts'][1:])]
        sources=[(oldfolder/phase,512,True) for phase in spec['original_phases']]+[(root/'evidence/l169'/phase,None,False) for phase in ['pilot-1','remaining-complete']]
        for folder,fixed_k,reused in sources:
            receipt=json.loads((folder/'receipt.json').read_text())
            expected_manifest=oldfolder/'input-manifest.json' if reused else root/'evidence/l169/input-manifest.json'
            if receipt['input_manifest_sha256']!=hashlib.sha256(expected_manifest.read_bytes()).hexdigest():raise ValueError('Receipt input differs')
            for r in receipt['records']:
                if not reused and r['database']!=db:continue
                k=fixed_k if reused else r['context'];arm=r['arm'];seed=r['seed'];identity=(db,arm,k,seed)
                if type(seed) is not int or type(k) is not int or arm not in manifest['arms'] or k not in manifest['contexts'] or seed not in range(10) or identity in seen:raise ValueError('Invalid or duplicate run')
                if not reused and k==512:raise ValueError('512 context must be authenticated reused evidence')
                seen.add(identity);path=folder/(f'{arm}-{seed}.npz' if reused else f'{db}-{arm}-{k}-{seed}.npz')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Receipt prediction hash differs')
                if not reused and 'origin_phase' in r:
                    origin=root/'evidence/l169'/r['origin_phase']/r['filename']
                    if hashlib.sha256(origin.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Original shard hash differs')
                    record_path=origin.with_suffix('.json')
                    if hashlib.sha256(record_path.read_bytes()).hexdigest()!=r['origin_record_sha256']:raise ValueError('Original per-run receipt differs')
                    original_record=json.loads(record_path.read_text())
                    if any(r.get(key)!=value for key,value in original_record.items()):raise ValueError('Assembled record changed original fields')
                with np.load(path,allow_pickle=False) as out:
                    if not np.array_equal(out['support_keys'],train[data['support_'+str(k)][seed]]):raise ValueError('Support identities differ')
                    if len(out['keys'])!=len(keys) or len(set(map(tuple,out['keys'])))!=len(keys):raise ValueError('Incomplete prediction keys')
                    if dict(zip(map(tuple,out['keys']),out['label']))!=dict(zip(map(tuple,keys),labels)):raise ValueError('Wrong keyed labels')
                    auc=auc_fn(keys,labels,out['keys'],out['probability'])
                    if abs(auc-r['auc'])>1e-12:raise ValueError('Metric receipt differs')
                rows.append(dict(database=db,arm=arm,context=k,seed=seed,auc=auc,evidence='REUSED' if reused else 'FRESH'))
                raw_count+=len(keys);fresh_count+=not reused;reused_count+=reused
        for arm in manifest['arms']:
            curve=curve_fn([r for r in rows if r['database']==db and r['arm']==arm]);curves[db+'/'+arm]=curve
            for level in curve['levels']:
                k=level['context'];target=manifest['paper_targets'][str(k)][arm][db];delta=level['mean']-target
                comparisons.append(dict(database=db,arm=arm,context=k,mean=level['mean'],sample_sd=level['sample_sd'],paper=target,delta=delta,status='CLOSE' if abs(delta)<=manifest['descriptive_tolerance'] else 'OUTSIDE_TOLERANCE',evidence='REUSED' if k==512 else 'FRESH'))
        for k in manifest['contexts']:
            a={r['seed']:r['auc'] for r in rows if r['database']==db and r['arm']=='RDBPFN' and r['context']==k}
            b={r['seed']:r['auc'] for r in rows if r['database']==db and r['arm']=='TabICLv1.1' and r['context']==k}
            delta=np.array([a[s]-b[s] for s in range(10)])
            pairs[db+'/'+str(k)]=dict(per_seed=delta.tolist(),mean=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum()))
    expected={(d['database'],a,k,s) for d in manifest['experiments'] for a in manifest['arms'] for k in manifest['contexts'] for s in range(10)}
    if seen!=expected or fresh_count!=240 or reused_count!=60:raise ValueError('Incomplete selected experiment')
    recovery=json.loads((root/'evidence/l169/remaining-complete/receipt.json').read_text())
    sentinel_checks=[]
    for record in json.loads((root/'evidence/l169/tail-1/receipt.json').read_text())['records']:
        if record['context']!=64:continue
        name=record['filename']
        with np.load(root/'evidence/l169/remaining-2'/name) as a,np.load(root/'evidence/l169/tail-1'/name) as b:
            for field in ['keys','label','support_keys']:
                if not np.array_equal(a[field],b[field]):raise ValueError('Sentinel identity mismatch')
            delta=float(np.max(np.abs(a['probability']-b['probability'])))
            aa=auc_fn(a['keys'],a['label'],a['keys'],a['probability']);bb=auc_fn(b['keys'],b['label'],b['keys'],b['probability'])
            sentinel_checks.append(dict(database=record['database'],arm=record['arm'],max_probability_difference=delta,auc_difference=bb-aa,exact_probability_match=delta==0))
    if len(sentinel_checks)!=6:raise ValueError('Incomplete sentinel diagnostics')
    for a,b in zip(sentinel_checks,recovery['sentinel_results']):
        if any(abs(a[k]-b[k])>1e-12 for k in ['max_probability_difference','auc_difference']) or a['arm']!=b['arm'] or a['database']!=b['database']:raise ValueError('Sentinel diagnostic differs')
    return dict(experiment=manifest['experiment'],status='COMPLETE_SELECTED_CONTEXT_SWEEP',fresh_runs=fresh_count,reused_runs=reused_count,all_predictions_checked=raw_count,
                exact_repeatability='PASS' if all(x['exact_probability_match'] for x in sentinel_checks) else 'FAIL_TABICL',sentinel_checks=sentinel_checks,
                fresh_predictions=183240,reused_predictions=45810,curves=curves,comparisons=comparisons,paired_models=pairs,support_overlap=support_overlaps,
                claim=claim_fn('context',True,5,False),hashed_files=len(pins['files']),
                historical_identity='NOT_ESTABLISHED',historical_availability='NOT_ESTABLISHED',original_dfs_regeneration='NOT_RUN',exact_target_schema_exclusion='NOT_ESTABLISHED',
                pretraining_scaling_law='NOT_ESTABLISHED',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import json,signal
    from pathlib import Path
    from relkit.transfer_l167 import keyed_auc
    from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
    signal.alarm(600);P=Path(__file__).resolve().parent;E=P/'evidence/l169'
    result=audit169(P,json.loads((E/'audit-manifest.json').read_text()),keyed_auc,sample_context,scaling_curve,scaling_claim)
    (E/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['all_predictions_checked'])
