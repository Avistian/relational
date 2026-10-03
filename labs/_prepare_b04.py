"""Freeze the separate course diagnostic before model execution."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
P=Path(__file__).resolve().parent;E=P/'evidence/b04'
if (E/'diagnostic-protocol.json').exists():raise SystemExit('Refusing to overwrite frozen protocol')
d=load_breast_cancer();train,test=train_test_split(np.arange(len(d.target)),test_size=64,random_state=0,stratify=d.target)
support=np.random.default_rng(4).permutation(train).tolist();test=test.tolist()
mask=np.random.default_rng(404).random(d.data.shape)<.15
raw=dict(dataset='sklearn.datasets.load_breast_cancer',X=d.data.tolist(),y=d.target.tolist(),feature_names=d.feature_names.tolist(),support_order=support,query_ids=test,missing_mask=mask.tolist())
b=(json.dumps(raw,sort_keys=True,separators=(',',':'))+'\n').encode();(E/'inputs.json').write_bytes(b)
configs=[dict(name=f'clean-{n}',support_n=n,missing=False,indicators=False,query_batch=64) for n in [32,128,384]]
configs += [dict(name='batch-128',support_n=128,missing=False,indicators=False,query_batch=16),dict(name='mean-128',support_n=128,missing=True,indicators=False,query_batch=64),dict(name='indicator-128',support_n=128,missing=True,indicators=True,query_batch=64)]
settings=dict(n_estimators=1,norm_methods='none',feat_shuffle_method='none',class_shuffle_method='none',outlier_threshold=4.0,softmax_temperature=.9,average_logits=True,support_many_classes=False,batch_size=1,kv_cache=False,allow_auto_download=False,checkpoint_version='tabicl-classifier-v2-20260212.ckpt',device='cpu',use_amp=False,use_fa3=False,offload_mode=False,random_state=0,n_jobs=1,verbose=False)
p=dict(name='B04-COURSE-CONTEXT-MISSINGNESS',inputs_sha256=hashlib.sha256(b).hexdigest(),checkpoint=json.loads((E/'checkpoint.json').read_text()),source_commit='0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3',configs=configs,settings=settings,query_batch_atol=1e-6,query_batch_rtol=0,torch_threads=1,fit_selection='none; no HPO or score-dependent changes',missingness='independent Bernoulli .15 mask, NumPy default_rng(404), frozen before scoring; append flags for all 30 columns; means fit on each support subset only; all-missing fill 0',memory='fresh-process Linux ru_maxrss peak RSS MiB including imports/model/data, not per-layer allocation or GPU memory; one process per configuration',timing='one fit and prediction sequence per process; no warmed-repeat timing; includes preprocessing and checkpoint load in fit',forecast_rule='one complete clean-32 pilot, projected remaining runtime 20 times pilot wall seconds plus 120s reserve; admit only within remaining local budget',paper_relationship='Separate course diagnostic; one view replaces release default eight only in this explicitly separate lane; no paper parity or large-context guarantee')
(E/'diagnostic-protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Frozen six configurations, 384 prediction rows, nine predict calls; inputs',p['inputs_sha256'])
