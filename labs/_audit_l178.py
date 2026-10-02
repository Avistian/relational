"""Portable complete prediction replay and independently reconstructed task contract."""
def audit178(root,pins,score_fn,gate_fn,select_fn):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    root=Path(root);E=root/'evidence/l178'
    for name,digest in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen input: '+name)
    data=np.load(root/'evidence/l166/prepared.npz');keys=data['test_keys'];labels=data['y_test'];trainkeys=data['train_keys']
    expected={(a,s) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)};seen=set();rows=[];runs=[]
    for phase in ['pilot-2','full-1']:
        folder=root/'evidence/l166'/phase;receipt=json.loads((folder/'receipt.json').read_text())
        for record in receipt['records']:
            arm,seed=record['arm'],record['seed'];identity=(arm,seed)
            if identity not in expected or identity in seen:raise ValueError('Duplicate/unexpected replay run')
            seen.add(identity);path=folder/(arm+'-'+str(seed)+'.npz')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Receipt hash mismatch')
            a=np.load(path);idx=data['support'][seed]
            state=int.from_bytes(hashlib.sha256(f'rel-f1-dfs-2:driver-dnf:{seed}'.encode()).digest()[:4],'big')
            original=np.random.default_rng(state).choice(len(trainkeys),512,replace=False)
            if not np.array_equal(idx,original) or not np.array_equal(a['support_keys'],trainkeys[idx]):raise ValueError('Changed published support identities')
            if not np.array_equal(a['keys'],keys) or not np.array_equal(a['label'],labels):raise ValueError('Different test keys/labels')
            value=score_fn(keys,labels,a['keys'],a['probability'])
            complemented=score_fn(keys,1-labels,a['keys'],1-a['probability'].astype('float64'))
            if abs(value-record['auc'])>1e-12 or abs(value-complemented)>1e-12:raise ValueError('Metric or complement invariance failure')
            rows.append(dict(arm=arm,seed=seed,auc=value));runs.append(dict(arm=arm,seed=seed,auc=value))
    if seen!=expected or len(keys)!=702:raise ValueError('Incomplete selected published replay')
    models={};old=json.loads((root/'evidence/l166/report.json').read_text())
    for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
        values=[r['auc'] for r in sorted(rows,key=lambda x:x['seed']) if r['arm']==arm]
        if values!=old['models'][arm]['per_seed']:raise ValueError('Replay differs from original report')
        models[arm]=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),per_seed=values,paper=old['models'][arm]['paper'])
    # Binary-search event intervals implement (cutoff, cutoff+30days], without source SQL.
    raw=pd.read_parquet(root/'evidence/l171/db/results.parquet');raw['ns']=raw.date.astype('datetime64[ns]').astype('int64')
    groups={int(k):v.sort_values('ns') for k,v in raw.groupby('driverId')};raw_reports={};raw_keys={};raw_labels={}
    for split in ['train','validation','test']:
        a=np.load(E/'released-task'/(split+'.npz'));k=np.column_stack([a['driverId'],a['date']]);n=len(k)
        if len(set(map(tuple,k)))!=n:raise ValueError('Duplicate raw task key')
        recomputed=np.empty(n,dtype=int);window_counts=np.empty(n,dtype=int);alternate=np.empty(n,dtype=int)
        for i,(driver,cut) in enumerate(k):
            g=groups[int(driver)];t=g.ns.to_numpy();out=(g.statusId.to_numpy()!=1)
            start=np.searchsorted(t,cut,side='right');end=np.searchsorted(t,cut+30*86400*10**9,side='right');end60=np.searchsorted(t,cut+60*86400*10**9,side='right')
            window_counts[i]=end-start;recomputed[i]=int(out[start:end].any());alternate[i]=int(out[start:end60].any())
        if (window_counts==0).any():raise ValueError('Unsupported outcome-free task row')
        if not np.array_equal(recomputed,1-a['did_not_finish']):raise ValueError('Raw 30-day labels are not exact release complements')
        raw_keys[split]=k;raw_labels[split]=recomputed
        raw_reports[split]=dict(rows=n,released_positive=int(a['did_not_finish'].sum()),dnf_positive=int(recomputed.sum()),complement_matches=int((recomputed==1-a['did_not_finish']).sum()),horizon_30_vs_60_disagreements=int((recomputed!=alternate).sum()))
    if any(set(map(tuple,raw_keys[a]))&set(map(tuple,raw_keys[b])) for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Split overlap')
    if not np.array_equal(raw_keys['train'],trainkeys) or not np.array_equal(raw_keys['test'],keys):raise ValueError('Raw/replay keys differ')
    supports=[]
    for seed in range(3):
        state=int.from_bytes(hashlib.sha256(f'rel-f1-dfs-2:driver-dnf:{seed}'.encode()).digest()[:4],'big')
        idx=np.random.default_rng(state).choice(len(trainkeys),1024,replace=False)
        if not (trainkeys[idx,1]+30*86400*10**9<raw_keys['validation'][:,1].min()).all():raise ValueError('Support outcomes unavailable before validation')
        supports.append(dict(seed=seed,rows=1024,keys_sha256=hashlib.sha256(np.ascontiguousarray(trainkeys[idx]).tobytes()).hexdigest(),dnf_positive=int(raw_labels['train'][idx].sum())))
    gradient=json.loads((E/'gradient-preflight.json').read_text());independent=json.loads((E/'gradient-independent.json').read_text());dfs=json.loads((E/'fastdfs-preflight.json').read_text())
    frame=pd.read_parquet(E/'gradient-input.parquet');cols=gradient['numerical_columns'];values=frame[cols].to_numpy(dtype=float)
    if not (frame.date<pd.Timestamp(gradient['query']['cutoff'])).all():raise ValueError('Future gradient probe row')
    if not frame.driverId.eq(gradient['query']['driverId']).all():raise ValueError('Different probe owner')
    # Independent arithmetic failure witness from raw cells. This is not a full backward rerun.
    normalized=(values-np.nanmean(values,axis=0))/(np.nanstd(values,axis=0)+1e-6)
    witness=int((~np.isfinite((normalized*np.where(np.isnan(normalized),0.,1.)).sum(axis=0))).sum())*128
    if witness!=gradient['nonfinite_gradient_elements'] or witness!=independent['source_linear_encoder_nonfinite']:raise ValueError('Gradient witness differs')
    if independent['early_imputation_control_nonfinite']!=0:raise ValueError('Invalid diagnostic control')
    for c in cols:
        if int(frame[c].isna().sum())!=gradient['missing_cells'][c]:raise ValueError('Missingness count differs')
    dbdigest=hashlib.sha256(json.dumps({n:d for n,d in pins['files'].items() if n.startswith('evidence/l171/db/')},sort_keys=True).encode()).hexdigest()
    common=dict(task='driver-dnf',horizon_days=30,positive='DNF',query_digest=hashlib.sha256(keys.tobytes()).hexdigest(),database_digest=dbdigest,support_digest=hashlib.sha256(json.dumps(supports,sort_keys=True).encode()).hexdigest(),target_history='NONE',temporal_audit='NOT_CHECKED',preprocessing_audit='NOT_CHECKED',selection='VALIDATION_ONLY',cost_ceiling=1.5)
    contracts={a:dict(common) for a in ['RDBPFN','RelGNN','RDBLearn']};contracts['RelGNN']['training_health']=gradient['status']
    decision=gate_fn(contracts)
    try:select_fn([],['0.001','0.005'],[0,1,2],10)
    except ValueError:selection='REJECTED_INCOMPLETE_TUNING_GRID'
    else:raise ValueError('Missing tuning runs were accepted')
    return dict(experiment='L178 Matched-Information F1 Comparison',fresh=dict(status=decision['status'],gate=decision,contracts=contracts,model_runs=0,predictions=0,training_selection=selection,gradient=gradient,independent_gradient=independent,fastdfs_preflight=dfs,complete_feature_regeneration='NOT_RUN',complete_temporal_audit='NOT_RUN',pinned_cuda_runtime='NOT_RUN',cloud_usd=0),replay=dict(status='COMPLETE_SELECTED_PUBLISHED_REPLAY',runs=30,predictions=21060,queries=702,models=models),raw_task=dict(status='PASS',horizon_days=30,positive='statusId != 1 in (cutoff, cutoff+30 days]',source_rows=len(raw),splits=raw_reports),supports=supports,
                boundaries=dict(fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_availability='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE'))

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.comparison_l178 import keyed_auc,comparison_gate,select_validation
    P=Path(__file__).resolve().parent;E=P/'evidence/l178'
    report=audit178(P,json.loads((E/'audit-input-manifest.json').read_text()),keyed_auc,comparison_gate,select_validation)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
