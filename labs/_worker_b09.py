"""One fresh process/model/split. Timings exclude download, include wrapper work."""
import os,time,sys,json,resource,random,hashlib
from pathlib import Path
START=time.perf_counter();P=Path(__file__).resolve().parent
sys.path[:0]=['/tmp/b09-deps','/tmp/b09-src/tabfm','/tmp/b09-src/exaone/src','/tmp/b09-src/nori/src','/tmp/b09-src/nori/libs/synthefy/src']
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from relkit.cost_b09 import aligned_rmse
name=sys.argv[1];seed=int(sys.argv[2]);E=P/'evidence/b09';out=E/f'{name}-{seed}.json'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
x,y=load_diabetes(return_X_y=True,scaled=False);x=x.astype('float32');y=y.astype('float32')
train,test=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed)
packet=P/f'data/b09/split-{seed}.npz'
if not packet.exists():np.savez(packet,x=x,y=y,train=train,test=test)
weights=Path('/tmp/b09-weights');t=time.perf_counter()
if name=='tabfm':
 from tabfm import TabFMRegressor,tabfm_v1_0_0_pytorch
 model=tabfm_v1_0_0_pytorch.load(model_type='regression',checkpoint_path=str(weights/'tabfm/regression'),device='cpu',dtype=torch.float32,use_cache=False)
 estimator=TabFMRegressor(model=model,n_estimators=1,norm_methods=['none'],feat_shuffle_method='none',random_state=seed,use_amp=False,cache_context=False)
 config=estimator.get_params();config['model']='pinned checkpoint';config={k:str(v) for k,v in config.items()}
elif name=='exaone':
 from exaonetabular import EXAONETabularRegressor
 estimator=EXAONETabularRegressor.from_pretrained(weights=str(weights/'exaone/exaone-tabular-regressor-v1_default.safetensors'),device='cpu',compute_dtype='float32',seed=seed)
 config=str(estimator.manifest)
else:
 from synthefy_nori import NoriRegressor
 estimator=NoriRegressor(model_path=str(weights/'nori/nori.pt'),device='cpu')
 config={k:str(v) for k,v in estimator.get_params().items()}
load_s=time.perf_counter()-t
t=time.perf_counter();estimator.fit(x[train],y[train]);fit_s=time.perf_counter()-t
t=time.perf_counter();pred=np.asarray(estimator.predict(x[test])).reshape(-1);first_s=time.perf_counter()-t
for _ in range(2):estimator.predict(x[test])
times=[];max_diff=0.
for _ in range(10):
 t=time.perf_counter();p=np.asarray(estimator.predict(x[test])).reshape(-1);times.append(time.perf_counter()-t);max_diff=max(max_diff,float(np.max(np.abs(p-pred))))
np.savez(E/f'{name}-{seed}-predictions.npz',ids=test,predictions=pred)
r=dict(model=name,seed=seed,status='COMPLETE',n_support=len(train),n_query=len(test),rmse=aligned_rmse(test,y[test],test,pred),load_s=load_s,fit_s=fit_s,first_predict_s=first_s,warm_s=times,warm_median_s=float(np.median(times)),warm_p90_s=float(np.quantile(times,.9)),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,repeat_max_abs_diff=max_diff,config=config,wall_s=time.perf_counter()-START,device='CPU',dtype='float32',threads=4,python=sys.version,torch=torch.__version__,numpy=np.__version__,input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest())
out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
