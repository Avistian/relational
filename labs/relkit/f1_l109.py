"""L109: full RelBench v1.1.0 driver-position label and heuristic reconstruction."""
# %% PROVIDED · authenticate two public archives before parsing
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SOURCE_COMMIT='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639'
ARCHIVES={
 'db.zip':('https://relbench.stanford.edu/download/rel-f1/db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'),
 'driver-position.zip':('https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')}
ARMS=['global_zero','global_mean','global_median','entity_mean','entity_median']
TARGETS={'val':[11.083,4.334,4.136,7.181,7.114],'test':[11.926,4.513,4.399,8.501,8.519]}
VAL=pd.Timestamp('2005-01-01');TEST=pd.Timestamp('2010-01-01');HORIZON=pd.Timedelta(days=60)

def load_inputs(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    frames={};metadata={}
    for name,(url,digest) in ARCHIVES.items():
        path=cache/name
        raw=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Archive identity mismatch: '+name)
        if not path.exists():path.write_bytes(raw)
        frames[name]={}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.namelist():
                if not member.endswith('.parquet'):continue
                table=pq.read_table(io.BytesIO(archive.read(member)));key=Path(member).stem
                frames[name][key]=table.to_pandas()
                if name=='db.zip':
                    metadata[key]={k:json.loads(table.schema.metadata[k.encode()]) for k in ['pkey_col','time_col','fkey_col_to_pkey_table']}
    return frames['db.zip'],frames['driver-position.zip'],metadata

# %% PROVIDED · reconstruct the source's query schedule, including empty windows

def schedule(tables,metadata,split):
    timed=[df[metadata[name]['time_col']] for name,df in tables.items() if metadata[name]['time_col'] is not None]
    if split in ['train','val']:timed=[x[x<=TEST] for x in timed]
    minimum=min(x.min() for x in timed);maximum=max(x.max() for x in timed)
    if split=='train':return pd.date_range(VAL-HORIZON,minimum,freq=-HORIZON)
    if split=='val':return pd.date_range(VAL,min(VAL+39*HORIZON,TEST-HORIZON),freq=HORIZON)
    if split=='test':return pd.date_range(TEST,min(TEST+39*HORIZON,maximum-HORIZON),freq=HORIZON)
    raise ValueError(split)

# %% PROVIDED · independent pandas reconstruction of DriverPositionTask.make_table

def regenerate(tables,metadata,split):
    """Preserve the released future-conditional population; no silent protocol repair.

    The source's lower-only one-year activity predicate is redundant for a driver
    with a result in (t,t+60d]. The selected future result itself satisfies it.
    We keep the explicit predicate to expose the source behavior.
    """
    results=tables['results']
    if split!='test':results=results[results.date<=TEST]
    valid_drivers=set(tables['drivers'].driverId);parts=[]
    for t in schedule(tables,metadata,split):
        future=results[(results.date>t)&(results.date<=t+HORIZON)]
        active=set(results.loc[results.date>t-pd.DateOffset(years=1),'driverId'])
        future=future[future.driverId.isin(valid_drivers & active)]
        agg=future.groupby('driverId',as_index=False).positionOrder.mean().rename(columns={'positionOrder':'position'})
        agg.insert(0,'date',t);parts.append(agg)
    return pd.concat(parts,ignore_index=True)

# %% PROVIDED · deterministic estimators, no optimizer, search or seeds

def baseline_predict(fit,query,arm):
    if arm=='global_zero':return np.zeros(len(query))
    if arm=='global_mean':return np.full(len(query),fit.position.mean())
    if arm=='global_median':return np.full(len(query),fit.position.median())
    if arm in ['entity_mean','entity_median']:
        estimates=fit.groupby('driverId').position.agg(arm.split('_')[1])
        return query.driverId.map(estimates).fillna(0).to_numpy(dtype=float)
    raise ValueError(arm)

# %% PROVIDED · full row-level comparisons and reproducible quantitative evidence

def run_reproduction(cache,output):
    tables,tasks,metadata=load_inputs(cache);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    regenerated={};audits={}
    for split,released in tasks.items():
        gen=regenerate(tables,metadata,split)
        assert not gen.duplicated(['date','driverId']).any()
        assert not released.duplicated(['date','driverId']).any()
        aligned=released[['date','driverId']].merge(gen,on=['date','driverId'],how='left',validate='one_to_one')
        assert len(gen)==len(released) and aligned.position.notna().all(),'Query populations differ'
        error=float(np.max(np.abs(aligned.position.to_numpy()-released.position.to_numpy())))
        assert error<=1e-12,'Regenerated labels differ'
        regenerated[split]=aligned
        # Future-conditional source population vs bounded prior activity: diagnostic only.
        inactive=0
        for t,group in released.groupby('date'):
            prior=tables['results'];prior=prior[(prior.date>t-pd.DateOffset(years=1))&(prior.date<=t)]
            inactive+=int((~group.driverId.isin(prior.driverId)).sum())
        audits[split]={'rows':len(gen),'scheduled_windows':len(schedule(tables,metadata,split)),'nonempty_windows':int(released.date.nunique()),'max_cached_label_error':error,'queries_without_prior_year_result':inactive}
        aligned.to_csv(out/(split+'-regenerated.csv'),index=False)
    assert (regenerated['train'].date+HORIZON<=VAL).all()
    assert (regenerated['val'].date+HORIZON<=TEST).all()
    predictions={};records=[]
    for split in ['val','test']:
        fit=regenerated['train'] if split=='val' else pd.concat([regenerated['train'],regenerated['val']],ignore_index=True)
        query=regenerated[split][['date','driverId']];truth=regenerated[split].position.to_numpy()
        for arm,target in zip(ARMS,TARGETS[split]):
            pred=baseline_predict(fit,query,arm);mae=float(np.abs(pred-truth).mean())
            records.append({'split':split,'arm':arm,'fit_rows':len(fit),'evaluation_rows':len(query),'mae':mae,'paper_mae':target,'gap':abs(mae-target),'verdict':'MATCH' if abs(mae-target)<=.0005 else 'FAIL'})
            predictions[split+'_'+arm]=pred
        predictions[split+'_target']=truth
        predictions[split+'_driverId']=query.driverId.to_numpy()
        predictions[split+'_date_ns']=query.date.to_numpy(dtype='datetime64[ns]').astype('int64')
    np.savez_compressed(out/'predictions.npz',**predictions)
    schema={name:{**meta,'rows':len(tables[name]),'availability':'NOT_RECORDED','version_history':'NOT_RECORDED'} for name,meta in metadata.items()}
    (out/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    report={'status':'COMPLETE','experiment':'RelBench v1 Table 4: rel-f1/driver-position, five heuristics, validation and test',
      'source_commit':SOURCE_COMMIT,'archives':{k:{'url':v[0],'sha256':v[1]} for k,v in ARCHIVES.items()},'labels':audits,'results':records,
      'database_tables':len(tables),'database_rows':sum(map(len,tables.values())),'tolerance':.0005,'seed_policy':'Deterministic; seed repeats do not estimate uncertainty',
      'historical_execution_identity':'NOT_ESTABLISHED','full_paper_reproduction':'NOT_ESTABLISHED','real_availability_history':'NOT_AVAILABLE',
      'unrun':['Raw Kaggle-to-database reconstruction','LightGBM and RDL columns','Other tasks and datasets']}
    (out/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

# %% PROVIDED · turn the released schema into an explicitly scoped snapshot graph

def graph_snapshot(tables,metadata,cutoff):
    """Typed PK nodes and FK edges; fixed query cutoff, no feature encoding/training.

    Undated tables follow the release's always-eligible assumption. This cannot
    establish actual knowledge time. PK values remain stable across filtering.
    The returned arrays hold IDs, not renumbered row offsets. Missing endpoints
    suppress an edge and increment an audit counter; null FKs create no edge.
    """
    visible={};nodes={};edges={};dropped=0
    for name,frame in tables.items():
        meta=metadata[name];clock=meta['time_col']
        visible[name]=frame if clock is None else frame[frame[clock]<=cutoff]
        nodes[name]=visible[name][meta['pkey_col']].to_numpy(dtype=np.int64)
    for name,frame in visible.items():
        meta=metadata[name]
        for fk,parent in meta['fkey_col_to_pkey_table'].items():
            known=frame[fk].notna();valid=known & frame[fk].isin(nodes[parent])
            dropped+=int((known & ~valid).sum())
            pairs=frame.loc[valid,[meta['pkey_col'],fk]].to_numpy(dtype=np.int64)
            if len(pairs):pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
            edges[(name,fk,parent)]=pairs
    return {'nodes':nodes,'edges':edges,'dropped_dangling':dropped}
