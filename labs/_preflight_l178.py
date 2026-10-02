"""Execute actual FastDFS before admitting the matched three-model experiment."""
import copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
import fastdfs
P=Path(__file__).resolve().parent;E=P/'evidence/l178'

def database():
    tables={};pks={};times={};fks=[]
    for path in sorted((P/'evidence/l171/db').glob('*.parquet')):
        table=pq.read_table(path);meta=table.schema.metadata;name=path.stem;df=table.to_pandas()
        pks[name]=json.loads(meta[b'pkey_col']);t=json.loads(meta[b'time_col'])
        if t:times[name]=t
        for c,other in json.loads(meta[b'fkey_col_to_pkey_table']).items():fks.append((name,c,other,None))
        if name=='races' and 'time' in df and not pd.api.types.is_float_dtype(df.time):df['time']=pd.to_timedelta(df.time).dt.total_seconds()
        tables[name]=df
    fks=[(a,b,c,pks[c]) for a,b,c,_ in fks]
    return tables,pks,times,fks

def run():
    tables,pks,times,fks=database();config=fastdfs.DFSConfig(max_depth=2,use_cutoff_time=True,engine='dfs2sql')
    train=np.load(E/'released-task/train.npz');test=np.load(E/'released-task/test.npz');val=np.load(E/'released-task/validation.npz')
    # Retain the existing stable support convention and explicit three seed schedule.
    supports=[]
    for seed in range(3):
        key=f'rel-f1-dfs-2:driver-dnf:{seed}'
        state=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')
        supports.append(np.random.default_rng(state).choice(len(train['driverId']),1024,replace=False))
    np.savez_compressed(E/'support-schedule.npz',indices=supports)
    # A real early training query with both history and later results, chosen without metrics.
    candidate=[]
    r=tables['results']
    for i in supports[0]:
        date=pd.Timestamp(int(train['date'][i]));driver=int(train['driverId'][i]);rows=r[r.driverId==driver]
        if (rows.date<date).sum()>10 and (rows.date>date).sum()>10:candidate.append((int(i),driver,date));break
    i,driver,date=candidate[0]
    target=pd.DataFrame({'driverId':[driver],'date':[date]})
    def compute(data):
        rdb=fastdfs.create_rdb(data,name='l178-f1',primary_keys=pks,foreign_keys=fks,time_columns=times)
        return fastdfs.compute_dfs_features(rdb,target.copy(),{'driverId':'drivers.driverId'},'date',config)
    start=time.perf_counter();baseline=compute(copy.deepcopy(tables));seconds=time.perf_counter()-start
    changed=copy.deepcopy(tables);counts={}
    for table,col in times.items():
        df=changed[table];mask=df[col]>=date;counts[table]=int(mask.sum())
        for name in df.select_dtypes(include='number').columns:
            if name==pks[table] or any(a==table and b==name for a,b,_,_ in fks):continue
            df.loc[mask,name]=999999
    mutated=compute(changed)
    if set(baseline.columns)!=set(mutated.columns):raise ValueError('Future intervention changes feature schema')
    diffs=[]
    for c in baseline:
        a=baseline[c].iloc[0];b=mutated[c].iloc[0]
        if not ((pd.isna(a) and pd.isna(b)) or a==b):diffs.append(dict(feature=c,before=str(a),after=str(b)))
    result=dict(status='PASS' if not diffs else 'FAIL_FUTURE_INTERVENTION',query=dict(driverId=driver,cutoff=str(date),train_index=i),feature_columns=len(baseline.columns),seconds=seconds,changed_future_rows=counts,changed_features=diffs,
                scope='One actual query and all numeric non-key future cells; diagnostic preflight, not complete temporal proof',feature_config=config.model_dump(),target_augmentation=False)
    (E/'fastdfs-preflight.json').write_text(json.dumps(result,indent=2)+'\n');baseline.to_parquet(E/'preflight-baseline.parquet');mutated.to_parquet(E/'preflight-mutated.parquet');print(json.dumps(result,indent=2))
if __name__=='__main__':
    try:run()
    except Exception:
        (E/'fastdfs-failure.txt').write_text(traceback.format_exc());raise
