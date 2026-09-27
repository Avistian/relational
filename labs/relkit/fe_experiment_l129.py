"""Complete released manual-feature experiment, with explicit reproducibility adapters.

SQL: snap-stanford/relbench-user-study 445bb7a3b1230f49f8e5890ae81754d3e365680f.
LightGBM search mirrors pytorch-frame 0.2.2 (MIT, license in sources/l129/frame).
No subsampling. Archive loader replaces unavailable historical staging endpoints.
"""
import hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from jinja2 import Template

KEYS=['driverId','date']

def load_archives(source):
    source=Path(source)
    for spec in json.loads((source/'data-manifest.json').read_text()):
        assert hashlib.sha256((source/spec['path']).read_bytes()).hexdigest()==spec['sha256']
    tables={};raw_counts={}
    with zipfile.ZipFile(source/'db.zip') as z:
        for name in z.namelist():
            if not name.endswith('.parquet'):continue
            table=pq.read_table(io.BytesIO(z.read(name)),use_threads=False)
            df=table.to_pandas();time_col=json.loads(table.schema.metadata[b'time_col'])
            raw_counts[Path(name).stem]=len(df)
            if time_col is not None:df=df[df[time_col]<=pd.Timestamp('2010-01-01')]
            tables[Path(name).stem]=df.reset_index(drop=True)
    with zipfile.ZipFile(source/'driver-position.zip') as z:
        labels={s:pq.read_table(io.BytesIO(z.read(f'driver-position/{s}.parquet')),use_threads=False).to_pandas() for s in ['train','val','test']}
    return tables,labels,raw_counts

def make_features(tables,labels,sql,threads=4):
    con=duckdb.connect();con.execute(f'SET threads={int(threads)}')
    for name,df in tables.items():
        con.register('incoming',df);con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM incoming')
    for split,df in labels.items():
        # Keep final test targets outside feature SQL entirely.
        if split=='test':df=df.drop(columns=['position'])
        con.register('incoming',df);con.execute(f'CREATE TABLE driver_position_{split} AS SELECT * FROM incoming')
        con.execute(Template(sql).render(set=split,subsample=0))
    features={s:con.sql(f'SELECT * FROM driver_position_{s}_feats').df() for s in labels}
    con.close()
    for split,df in features.items():
        assert len(df)==len(labels[split]) and not df.duplicated(KEYS).any()
        assert set(map(tuple,df[KEYS].to_numpy()))==set(map(tuple,labels[split][KEYS].to_numpy()))
        # SQL has no ORDER BY: freeze archive query order before seeded sampling.
        order=pd.MultiIndex.from_frame(labels[split][KEYS])
        features[split]=df.set_index(KEYS).loc[order].reset_index()
    return features

def materialize_features(features,stypes):
    from torch_frame.data import Dataset
    from torch_frame.gbdt import LightGBM
    from torch_frame import TaskType,Metric
    dataset=Dataset(features['train'],col_to_stype=stypes,target_col='position').materialize()
    model=LightGBM(TaskType.REGRESSION,metric=Metric.MAE)
    frames={s:dataset.tensor_frame if s=='train' else dataset.convert_to_tensor_frame(df) for s,df in features.items()}
    matrices={s:model._to_lightgbm_input(tf) for s,tf in frames.items()}
    return dataset,frames,matrices

def sample_parameters(trial):
    return dict(verbosity=-1,bagging_freq=1,
        max_depth=trial.suggest_int('max_depth',3,11),
        learning_rate=trial.suggest_float('learning_rate',1e-3,.1,log=True),
        num_leaves=trial.suggest_int('num_leaves',2,2**10),
        subsample=trial.suggest_float('subsample',.05,1.),
        colsample_bytree=trial.suggest_float('colsample_bytree',.05,1.),
        lambda_l1=trial.suggest_float('lambda_l1',1e-9,10.,log=True),
        lambda_l2=trial.suggest_float('lambda_l2',1e-9,10.,log=True),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',1,100),
        objective='regression_l1',metric='mae')

def tune_fe(matrices,seed,num_trials=10,rounds=2000,threads=4):
    import lightgbm as lgb
    import optuna
    from relkit.manual_fe_l129 import choose_trial
    train_x,train_y,cats=matrices['train'];val_x,val_y,_=matrices['val']
    train=lgb.Dataset(train_x,label=train_y,free_raw_data=False)
    val=lgb.Dataset(val_x,label=val_y,free_raw_data=False)
    trace=[]
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    def fit(params):
        return lgb.train(dict(params,num_threads=threads),train,num_boost_round=rounds,
            categorical_feature=cats,valid_sets=[val],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    def objective(trial):
        start=time.perf_counter();params=sample_parameters(trial);model=fit(params)
        pred=model.predict(val_x);score=float(np.mean(np.abs(pred-val_y)))
        trace.append(dict(number=trial.number,val_mae=score,best_iteration=model.best_iteration,params=params,seconds=time.perf_counter()-start))
        return score
    study.optimize(objective,n_trials=num_trials)
    selected=choose_trial(trace);assert selected==study.best_trial.number
    best=next(row for row in trace if row['number']==selected)
    model=fit(best['params']) # upstream refits the selected configuration
    assert abs(float(np.mean(np.abs(model.predict(val_x)-val_y)))-best['val_mae'])<1e-10
    return model,trace,selected
