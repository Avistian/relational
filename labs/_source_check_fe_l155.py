"""Run original pinned Frame trainer against the visible replay under one seeded search."""
import hashlib,importlib.util,inspect,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch,optuna,lightgbm
from torch_frame.gbdt import LightGBM
from torch_frame import TaskType,Metric
from relkit.fe_experiment_l129 import tune_fe,materialize_features
P=Path(__file__).resolve().parent;S=P/'sources/l129';O=P/'evidence/l155/fe'
assert hashlib.sha256(Path(inspect.getsourcefile(LightGBM)).read_bytes()).hexdigest()==hashlib.sha256((S/'frame/tuned_lightgbm.py').read_bytes()).hexdigest()
features={s:pd.read_parquet(O/f'{s}-features.parquet') for s in ['train','val','test']}
spec=importlib.util.spec_from_file_location('stypes',S/'inferred_stypes.py');st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
ds,frames,matrices=materialize_features(features,st.task_to_stypes['rel-f1-driver-position'])
a=np.load(O/'matrices.npz')
for split in frames:np.testing.assert_allclose(matrices[split][0].to_numpy(),a[split+'_x'],rtol=0,atol=0,equal_nan=True)
original_create=optuna.create_study;original_train=lightgbm.train;models=[]
def seeded_create(*args,**kwargs):
    kwargs['sampler']=optuna.samplers.TPESampler(seed=129);return original_create(*args,**kwargs)
def observed_train(params,*args,**kwargs):
    params=dict(params,num_threads=4);m=original_train(params,*args,**kwargs);models.append(m);return m
optuna.create_study=seeded_create;lightgbm.train=observed_train
start=time.perf_counter()
try:
    original=LightGBM(TaskType.REGRESSION,metric=Metric.MAE)
    original.tune(frames['train'],frames['val'],num_trials=2,num_boost_round=2000)
finally:optuna.create_study=original_create;lightgbm.train=original_train
pilot=lightgbm.Booster(model_file=str(O/'pilot/model.txt'))
errors={}
for split in ['val','test']:
    expected=original.predict(frames[split]).numpy();observed=pilot.predict(matrices[split][0]);error=float(np.max(np.abs(expected-observed)));assert error<1e-10;errors[split]=error
r=dict(status='PASS',original_training_calls=len(models),num_trials=2,round_cap=2000,seed=129,maximum_prediction_error=errors,matrix_values='EXACT',original_class_bytes='EXACT',adapters=['seed default TPE sampler','4 LightGBM threads'],seconds=time.perf_counter()-start)
(P/'_source_check_fe_l155_results.json').write_text(json.dumps(r,indent=2));print(r)
