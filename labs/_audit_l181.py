"""Portable complete-data baseline audit, with injected live learner functions."""
def audit181(packet,manifest,visible,baseline,score):
    import hashlib,json
    from pathlib import Path
    import numpy as np,pandas as pd
    packet=Path(packet)
    for name,digest in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed input: '+name)
    cfg=json.loads((packet/'config.json').read_text());gradient=json.loads((packet/'gradient-preflight.json').read_text())
    tasks={};predictions=0
    kinds=['global_zero','global_mean','global_median','entity_mean','entity_median']
    for task,spec in cfg['tasks'].items():
        tables={s:pd.read_parquet(packet/task/(s+'.parquet')) for s in ['train','val','test']}
        raw=pd.read_parquet(packet/'db'/(spec['table']+'.parquet'))
        for split,d in tables.items():
            lo,hi=map(pd.Timestamp,spec['split_ranges'][split]);rows=raw[(raw.date>lo)&(raw.date<=hi)&raw.position.notna()]
            expected=pd.DataFrame({'entity':rows[spec['key']].to_numpy(dtype=np.int64),'time':rows.date.astype('int64').to_numpy(),'y':rows.position.to_numpy(dtype=float)}).sort_values(['time','entity']).reset_index(drop=True)
            if not d.equals(expected) or len(d)!=spec['paper_counts'][split]:raise ValueError('Split/label mismatch: '+task+'/'+split)
        columns=raw.columns.tolist();actual=visible(columns,'position',spec['proxies'],True,'global')
        context=visible(columns,'position',spec['proxies'],False,'global')
        if actual!=context or set(actual)&set(['position',*spec['proxies']]):raise ValueError('Global masking violated')
        scores={};overlap={}
        for split in ['val','test']:
            fit=tables['train'] if split=='val' else pd.concat([tables['train'],tables['val']],ignore_index=True)
            q=tables[split];overlap[split]=len(set(q.entity)&set(fit.entity));scores[split]={}
            for kind in kinds:
                p=q[['entity','time']].copy();p['pred']=baseline(fit,q,kind)
                stored=pd.read_parquet(packet/task/(split+'-'+kind+'.parquet'))
                if not p.equals(stored):raise ValueError('Prediction mismatch: '+task+'/'+split+'/'+kind)
                scores[split][kind]=score(q,p);predictions+=len(p)
        tasks[task]=dict(counts=spec['counts'],fit_entity_overlap=overlap,removed_columns=['position',*spec['proxies']],retained_columns=actual,scores=scores)
    failed=[dict(task=r['task'],table=r['table'],rows=r['rows'],nonfinite_gradient_elements=r['nonfinite_gradient_elements']) for r in gradient['runs'] if r['nonfinite_gradient_elements']]
    return dict(experiment=cfg['experiment'],baseline_status='COMPLETE_SELECTED_BASELINE_REPRODUCTION',gnn_status='NOT_RUN_TRAINING_HEALTH_GATE' if failed else 'NOT_RUN',selected_experiment='INCOMPLETE',full_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',predictions=predictions,tasks=tasks,gradient_scope=gradient['scope'],gradient_failures=failed,cloud_usd=0,relgt_ac_reproduction='NOT_RUN_SOURCE_GAPS',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import hashlib,json
    from pathlib import Path
    import pandas as pd
    from relkit.autocomplete_l181 import visible_columns,baseline_predictions,keyed_scores
    P=Path(__file__).resolve().parent;E=P/'evidence/l181';packet=E/'packet'
    cfg=json.loads((packet/'config.json').read_text())
    for task in cfg['tasks']:
        train=pd.read_parquet(packet/task/'train.parquet');val=pd.read_parquet(packet/task/'val.parquet')
        for split in ['val','test']:
            fit=train if split=='val' else pd.concat([train,val],ignore_index=True);q=pd.read_parquet(packet/task/(split+'.parquet'))
            for kind in ['global_zero','global_mean','global_median','entity_mean','entity_median']:
                p=q[['entity','time']].copy();p['pred']=baseline_predictions(fit,q,kind);p.to_parquet(packet/task/(split+'-'+kind+'.parquet'),index=False)
    manifest=dict(files={str(p.relative_to(packet)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(packet.rglob('*')) if p.is_file()})
    (E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=audit181(packet,manifest,visible_columns,baseline_predictions,keyed_scores)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['baseline_status'],report['predictions'],report['gnn_status'])
