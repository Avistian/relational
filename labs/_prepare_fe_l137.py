"""Generate every SQL feature row and serialize the exact train-fitted matrices."""
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
from relkit.fe_experiment_l129 import load_archives,make_features,materialize_features
P=Path(__file__).resolve().parent;S=P/'sources/l129';O=P/'evidence/l137/fe';O.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();tables,labels,raw=load_archives(S)
sql=(S/'f1/driver-position/feats.sql').read_text();features=make_features(tables,labels,sql)
spec=importlib.util.spec_from_file_location('study_stypes',S/'inferred_stypes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
dataset,frames,matrices=materialize_features(features,m.task_to_stypes['rel-f1-driver-position'])
arrays={};identities={}
for split,df in features.items():
    df.to_parquet(O/f'{split}-features.parquet',index=False)
    labels[split].to_parquet(O/f'{split}-labels.parquet',index=False)
    x,y,cats=matrices[split]
    arrays[split+'_x']=x.to_numpy();arrays[split+'_y']=np.array([]) if y is None else y
    arrays[split+'_feature_id']=df.driverId.to_numpy(dtype='int64');arrays[split+'_feature_time']=df.date.astype('int64').to_numpy()
    arrays[split+'_query_id']=labels[split].driverId.to_numpy(dtype='int64');arrays[split+'_query_time']=labels[split].date.astype('int64').to_numpy();arrays[split+'_target']=labels[split].position.to_numpy()
    identities[split]=dict(rows=len(df),columns=len(df.columns),matrix_shape=list(x.shape))
arrays['categorical_indices']=np.array(cats)
np.savez_compressed(O/'matrices.npz',**arrays)
report=dict(status='PASS',raw_counts=raw,capped_counts={k:len(v) for k,v in tables.items()},splits=identities,feature_names={str(k):v for k,v in frames['train'].col_names_dict.items()},seconds=time.perf_counter()-start,versions={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ['torch','pytorch-frame','relbench','duckdb','pandas','numpy','lightgbm','optuna','jinja2']},archives=json.loads((S/'data-manifest.json').read_text()),matrix_sha256=hashlib.sha256((O/'matrices.npz').read_bytes()).hexdigest(),historical_data='NOT_ESTABLISHED: relbench0.2.0 staging endpoint404; archived v1 rows match released notebook dimensions')
(O/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
