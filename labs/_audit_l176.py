"""Portable complete-evidence audit; supplied live learner functions determine results."""
def audit176(root,pins,schedule_fn,score_fn,curve_fn):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    root=Path(root)
    for name,digest in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch: '+name)
    E=root/'evidence/l176';manifest=json.loads((E/'input-manifest.json').read_text())
    contexts=manifest['contexts'];seeds=manifest['seeds'];arms=manifest['arms'];rows=[];seen=set();count=0;support_checks=0;curves={};pairs={}
    expected={(s['database'],a,k,i) for s in manifest['experiments'] for a in arms for k in contexts for i in seeds}
    for spec in manifest['experiments']:
        db=spec['database'];data=np.load(E/(db+'.npz'));old=np.load(root/'evidence/l169'/(db+'.npz'))
        if hashlib.sha256((E/(db+'.npz')).read_bytes()).hexdigest()!=manifest['files'][db+'.npz']['sha256']:raise ValueError('Input manifest mismatch')
        for name in ['X_train','X_test','y_train','y_test','train_keys','test_keys']:
            if not np.array_equal(data[name],old[name],equal_nan=True):raise ValueError('Original task arrays changed')
        keys=data['test_keys'];labels=data['y_test'];train=data['train_keys']
        if len(keys)!=spec['test_rows'] or len(train)!=spec['train_rows'] or len(set(map(tuple,keys)))!=len(keys) or len(set(map(tuple,train)))!=len(train) or set(map(tuple,keys))&set(map(tuple,train)):raise ValueError('Invalid task identity')
        horizon=(60 if db=='rel-f1' else 365)*86400*10**9
        if not (train[:,1]+horizon<keys[:,1].min()).all():raise ValueError('Support outcome not ready')
        for seed in seeds:
            schedule=schedule_fn(len(train),contexts,spec['seed_key'].format(seed=seed))
            for k in contexts:
                idx=data['support_'+str(k)][seed]
                if not np.array_equal(idx,schedule[k]):raise ValueError('Support schedule differs')
                if len(set(idx))!=k or not np.array_equal(idx,data['support_1024'][seed,:k]):raise ValueError('Non-nested support')
                support_checks+=1
        for phase in ['pilot-1','remaining-1']:
            folder=E/phase;receipt=json.loads((folder/'receipt.json').read_text())
            if receipt['input_manifest_sha256']!=hashlib.sha256((E/'input-manifest.json').read_bytes()).hexdigest():raise ValueError('Worker manifest differs')
            for r in receipt['records']:
                if r['database']!=db:continue
                key=(db,r['arm'],r['context'],r['seed'])
                if key not in expected or key in seen:raise ValueError('Duplicate or invalid evaluation')
                seen.add(key);path=folder/r['filename']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Prediction receipt differs')
                out=np.load(path);idx=data['support_'+str(r['context'])][r['seed']]
                if not np.array_equal(out['support_keys'],train[idx]):raise ValueError('Different support identities')
                if len(out['keys'])!=len(keys) or dict(zip(map(tuple,out['keys']),out['label']))!=dict(zip(map(tuple,keys),labels)):raise ValueError('Different query labels')
                auc=score_fn(keys,labels,out['keys'],out['probability'])
                if abs(auc-r['auc'])>1e-12:raise ValueError('AUROC mismatch')
                rows.append(dict(database=db,arm=r['arm'],context=r['context'],seed=r['seed'],auc=auc));count+=len(keys)
        for arm in arms:curves[db+'/'+arm]=curve_fn([r for r in rows if r['database']==db and r['arm']==arm],contexts,seeds)
        for k in contexts:
            a={r['seed']:r['auc'] for r in rows if r['database']==db and r['arm']=='RDBPFN' and r['context']==k}
            b={r['seed']:r['auc'] for r in rows if r['database']==db and r['arm']=='TabICLv1.1' and r['context']==k}
            delta=np.array([a[s]-b[s] for s in seeds]);pairs[db+'/'+str(k)]=dict(per_seed=delta.tolist(),mean=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum()))
    if seen!=expected or count!=229050:raise ValueError('Incomplete experiment')
    return dict(experiment=manifest['experiment'],status='COMPLETE_SELECTED_COURSE_EXPERIMENT',fresh_evaluations=len(seen),predictions=count,nested_schedule_checks=support_checks,curves=curves,paired_models=pairs,whole_paper='NOT_RUN',fresh_pretraining='NOT_RUN',historical_availability='NOT_ESTABLISHED',original_dfs_regeneration='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
