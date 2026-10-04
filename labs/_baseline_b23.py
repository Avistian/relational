"""Ten fresh, fixed logistic fits: a course baseline, not a Table9 reproduction."""
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from importlib.metadata import version
P=Path(__file__).resolve().parent;E=P/'evidence/b23';O=E/'baseline-1'
if O.exists():raise RuntimeError('Immutable baseline attempt already exists')
O.mkdir();data=np.load('/tmp/l166-input/prepared.npz');records=[]
for seed in range(10):
 start=time.monotonic();idx=data['support'][seed]
 # keep_empty_features preserves source feature width, filling empty columns with zero.
 model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=42))
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);model.fit(data['X_train'][idx],data['y_train'][idx])
 prob=model.predict_proba(data['X_test'])[:,1]
 transformed=model[:-1].transform(data['X_test']);z=transformed@model[-1].coef_[0]+model[-1].intercept_[0]
 np.testing.assert_allclose(prob,1/(1+np.exp(-z)),rtol=1e-12,atol=1e-12)
 path=O/f'Logistic-{seed}.npz'
 np.savez_compressed(path,keys=data['test_keys'],label=data['y_test'],probability=prob,support_keys=data['train_keys'][idx],coef=model[-1].coef_,intercept=model[-1].intercept_,median=model[0].statistics_,mean=model[1].mean_,scale=model[1].scale_)
 row=dict(arm='Logistic',seed=seed,rows=702,support=512,auc=float(roc_auc_score(data['y_test'],prob)),seconds=time.monotonic()-start,iterations=int(model[-1].n_iter_[0]),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 records.append(row);print(row,flush=True)
(O/'receipt.json').write_text(json.dumps(dict(records=records,packages={n:version(n) for n in ['numpy','scikit-learn']},lane='COURSE_BASELINE',input_manifest_sha256=hashlib.sha256((E/'input-manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')
