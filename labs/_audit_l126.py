"""Independent SQL database census, metric scoring, identity and fault probes."""
import hashlib,io,json,sqlite3,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from relkit.beta_l126 import schema_audit,engagement_table,align_predictions,average_precision
from _check_l126 import check_schema,check_engagement,check_alignment,check_ap
P=Path(__file__).resolve().parent;E=P/'evidence/l126'
r=json.loads((E/'summary.json').read_text())
# Parse archive independently of RelBench and join every FK in SQLite.
tables={}
with zipfile.ZipFile(E/'f1-db.zip') as z:
    for name in z.namelist():
        if name.endswith('.parquet'):
            arrow=pq.read_table(io.BytesIO(z.read(name)));meta=arrow.schema.metadata
            tables[Path(name).stem]=(arrow.to_pandas(),{k.decode():json.loads(v) for k,v in meta.items() if k!=b'pandas'})
with sqlite3.connect(':memory:') as conn:
    for name,(df,meta) in tables.items():
        # SQL audit only needs identity columns, so nested/text objects are excluded.
        cols=set(meta['fkey_col_to_pkey_table'])
        if meta['pkey_col']:cols.add(meta['pkey_col'])
        if meta['time_col']:cols.add(meta['time_col'])
        df[sorted(cols)].to_sql(name+'_raw',conn,index=False)
        where=f''' WHERE "{meta['time_col']}" <= '2010-01-01 00:00:00' ''' if meta['time_col'] else ''
        conn.execute(f'CREATE VIEW "{name}" AS SELECT * FROM "{name}_raw"'+where)
        assert conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]==r['schema']['tables'][name]['rows']
    for rel in r['schema']['relations']:
        src,col,dest=rel['source'],rel['column'],rel['target'];pk=tables[dest][1]['pkey_col']
        counts=conn.execute(f'SELECT sum(s."{col}" IS NULL),sum(s."{col}" IS NOT NULL AND d."{pk}" IS NOT NULL),sum(s."{col}" IS NOT NULL AND d."{pk}" IS NULL) FROM "{src}" s LEFT JOIN "{dest}" d ON s."{col}"=d."{pk}"').fetchone()
        assert tuple(map(int,counts))==(rel['null'],rel['resolved'],rel['dangling'])
    predictions=pd.read_csv(E/'f1-predictions.csv');predictions.to_sql('predictions',conn,index=False)
    mae=conn.execute('SELECT avg(abs(score-target_for_offline_audit_only)) FROM predictions').fetchone()[0]
    assert abs(mae-r['test_mae'])<1e-12
    assert conn.execute('SELECT count(*) FROM (SELECT driverId,date FROM predictions GROUP BY driverId,date)').fetchone()[0]==760
# Independently parse all task rows and compute the predictor without the API.
with zipfile.ZipFile(E/'f1-task.zip') as z, sqlite3.connect(':memory:') as conn:
    task_rows=0
    for split in ['train','val','test']:
        frame=pq.read_table(io.BytesIO(z.read('driver-position/'+split+'.parquet'))).to_pandas()
        assert len(frame)==r['splits'][split] and not frame.duplicated(['driverId','date']).any()
        frame.to_sql(split,conn,index=False);task_rows+=len(frame)
    n=r['splits']['train']
    middle=conn.execute('SELECT position FROM train ORDER BY position LIMIT 1 OFFSET ?', (n//2,)).fetchone()[0]
    assert middle==r['training_median']
    val_mae=conn.execute('SELECT avg(abs(position-?)) FROM val',(middle,)).fetchone()[0]
    assert abs(val_mae-r['validation_mae'])<1e-12
runtime=json.loads((P/'_runtime_sources_l126.json').read_text())
import relbench
for name,digest in runtime['files'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
    relative=Path(name).relative_to('sources/l126/v1')
    assert (Path(relbench.__file__).parent/relative).read_bytes()==(P/name).read_bytes()
mutants={}
def forget_nulls(tabs):
    x=schema_audit(tabs)
    for e in x['relations']:e['dangling']+=e['null'];e['null']=0
    return x

def include_cutoff(u,e,ts):
    shifted=e.copy();shifted['time']=shifted.time+pd.Timedelta(nanoseconds=1)
    return engagement_table(u,shifted,ts)

def calendar_horizon(u,e,ts):return engagement_table(u,e,ts,horizon_days=731)
def future_eligibility(u,e,ts):
    x=engagement_table(u,e,ts)
    return x[x.contribution==1]

def untied_ap(y,s):
    y=np.asarray(y)[np.argsort(-np.asarray(s),kind='stable')]
    return float(np.sum((np.cumsum(y)/np.arange(1,len(y)+1))*y)/max(1,y.sum()))
for name,check,mutant in [('null_as_dangling',check_schema,forget_nulls),('cutoff_in_future',check_engagement,include_cutoff),('731_days',check_engagement,calendar_horizon),('future_based_eligibility',check_engagement,future_eligibility),('ignore_query_keys',check_alignment,lambda q,p,k:p.score.to_numpy()),('split_equal_scores',check_ap,untied_ap)]:
    try:check(mutant)
    except (AssertionError,ValueError):mutants[name]='REJECTED'
    else:raise AssertionError('Surviving mutation '+name)
manifest=json.loads((P/'_sources_l126.json').read_text())
for name,item in manifest['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==item['sha256']
report={'status':'PASS','independent_sql_tables':len(tables),'raw_archive_rows':sum(len(x[0]) for x in tables.values()),'independent_sql_rows':r['schema']['total_rows'],'independent_sql_relations':len(r['schema']['relations']),'independent_task_rows':task_rows,'independent_train_median':middle,'independent_validation_mae':val_mae,'independently_scored_predictions':len(predictions),'sql_mae':mae,'mutations':mutants,'beta_source_files':len(manifest['files']),'historical_full_contract':'NOT_RUN'}
(P/'_audit_l126_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
