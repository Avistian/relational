"""Complete archive tokenization; explicit fit admission differs from transformation."""
def load172(root, manifest):
    import hashlib,json,zipfile
    from pathlib import Path
    import pyarrow.parquet as pq
    root=Path(root)
    for name,digest in manifest['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe input path')
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
    schema=json.loads((root/'evidence/l172/schema.json').read_text())
    tables={}
    with zipfile.ZipFile(root/'evidence/l171/rel-f1-db.zip') as archive:
        members={n for n in archive.namelist() if n.endswith('.parquet')}
        if members!={'db/'+n+'.parquet' for n in schema}:raise ValueError('Incomplete table population')
        for name,spec in sorted(schema.items()):
            path=root/'evidence/l171/db'/(name+'.parquet')
            if path.read_bytes()!=archive.read('db/'+name+'.parquet'):raise ValueError('Archive member mismatch')
            table=pq.read_table(path);frame=table.to_pandas()
            meta={k:json.loads(table.schema.metadata[k.encode()]) for k in ['pkey_col','time_col','fkey_col_to_pkey_table']}
            if meta['time_col']!=spec['time_col']:raise ValueError('Clock mismatch')
            for c,cs in spec['columns'].items():
                if str(frame[c].dtype)!=cs['storage_dtype']:raise ValueError('Storage dtype mismatch')
                expected='primary_key' if c==meta['pkey_col'] else 'foreign_key' if c in meta['fkey_col_to_pkey_table'] else 'event_time' if c==meta['time_col'] else 'feature'
                if cs['role']!=expected:raise ValueError('Key role mismatch')
                if expected=='foreign_key' and cs['target_table']!=meta['fkey_col_to_pkey_table'][c]:raise ValueError('FK target mismatch')
            tables[name]=frame
    return tables,schema


def replay172(root,manifest,fit,encode,tokenize):
    import collections,hashlib,json
    import numpy as np
    import pandas as pd
    tables,schema=load172(root,manifest)
    result=dict(experiment='L172 Full F1 Schema-Tokenization Audit',status='COMPLETE_FULL_SNAPSHOT_TOKENIZATION',tables={},rows=0,cells=0,columns=0,cloud_usd=0,
        historical_availability='NOT_ESTABLISHED',whole_paper_reproduction='NOT_RUN',fresh_pretraining='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    for name,frame in tables.items():
        spec=schema[name];clock=spec['time_col']
        admitted=(frame[clock]<pd.Timestamp(spec['fit_cutoff'])).tolist() if clock else [False]*len(frame)
        tokens=tokenize(frame,spec['columns'],admitted)
        reordered=tokenize(frame[frame.columns[::-1]],spec['columns'],admitted)
        if tokens!=reordered:raise AssertionError('Column reorder changed named tokens')
        digest=hashlib.sha256(json.dumps(tokens,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        summaries={}
        for column,out in tokens['columns'].items():
            kind=out['fitted']['kind'];values=frame[column]
            # All rows are exercised with the mask on, including source-null values.
            erased=encode(values,out['fitted'],[True]*len(frame))
            neutral='' if kind in {'key','text'} else 0.
            assert erased['state']==['MASKED']*len(frame) and erased['payload']==[neutral]*len(frame)
            if kind in {'number','category'}:
                mutated=values.astype(object).copy()
                mutated.loc[~np.asarray(admitted,dtype=bool)]=123456789. if kind=='number' else '__FUTURE_ONLY__'
                assert fit(mutated,kind,admitted)==out['fitted'],'Heldout intervention changed fit'
            summaries[column]=dict(kind=kind,role=out['schema']['role'],fit_rows=out['fitted']['fit_rows'],fit_nonnull=out['fitted']['fit_nonnull'],states=dict(sorted(collections.Counter(out['state']).items())),
                fit_sha256=hashlib.sha256(json.dumps(out['fitted'],sort_keys=True,allow_nan=False).encode()).hexdigest())
        if name=='results':
            c='points';out=tokens['columns'][c]
            i=int(np.flatnonzero(~np.asarray(admitted,dtype=bool))[0]);key=frame['resultId'].iloc[i]
            result['worked_trace']=dict(table=name,column=c,row_index=i,resultId=str(key),date=str(frame['date'].iloc[i]),raw=float(frame[c].iloc[i]),mean=out['fitted']['mean'],scale=out['fitted']['scale'],state=out['state'][i],payload=out['payload'][i],fit_rows=sum(admitted))
        result['tables'][name]=dict(rows=len(frame),columns=len(frame.columns),cells=len(frame)*len(frame.columns),admitted_rows=sum(admitted),time_col=clock,availability='NOT_ESTABLISHED',token_sha256=digest,column_audit=summaries)
        result['rows']+=len(frame);result['cells']+=len(frame)*len(frame.columns);result['columns']+=len(frame.columns)
    result['checks']=dict(column_reordering='PASS_ALL_TABLES',all_cell_mask_erasure='PASS',heldout_fit_intervention='PASS_ALL_NUMBER_CATEGORY_COLUMNS')
    return result

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.tokenization_l172 import fit_column,encode_column,tokenize_table
    P=Path(__file__).resolve().parent
    report=replay172(P,json.loads((P/'evidence/l172/input-manifest.json').read_text()),fit_column,encode_column,tokenize_table)
    (P/'evidence/l172/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['rows'],'rows',report['columns'],'columns',report['cells'],'cells')
