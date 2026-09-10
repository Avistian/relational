"""Compare local equations to translated official GKM Cholesky/Schur algebra.

Translation of Gavin Cawley @krr/train.m and @rbf/evaluate.m (2007).
SPDX-License-Identifier: GPL-2.0-or-later. See sources/l059/GPL-2.0.txt.
This is NOT native MATLAB execution or optimizer/runtime parity.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from relkit.validation_audit_l059 import krr_fit,predict_krr,loo_residuals
ROOT=Path(__file__).resolve().parent

def source_algebra(x,y,lam,eta,query):
    # @rbf/evaluate.m scales inputs by sqrt(eta), then expands distances.
    a=x*np.sqrt(eta);b=query*np.sqrt(eta)
    k=np.exp(-(np.sum(a*a,1)[:,None]+np.sum(a*a,1)[None,:]-2*a@a.T))
    q=np.exp(-(np.sum(b*b,1)[:,None]+np.sum(a*a,1)[None,:]-2*b@a.T))
    r=np.linalg.cholesky(k+lam*np.eye(len(y))).T
    zeta=np.linalg.solve(r,np.linalg.solve(r.T,np.ones(len(y))))
    xi=np.linalg.solve(r,np.linalg.solve(r.T,y))
    delta=1/zeta.sum();bias=xi.sum()*delta;alpha=xi-zeta*bias
    ri=np.linalg.inv(r);diagonal=np.sum(ri**2,axis=1)-delta*zeta**2
    return q@alpha+bias,alpha/diagonal,bias

with threadpool_limits(1):
    manifest=json.loads((ROOT/'_sources_l059_v2.json').read_text())
    for row in manifest['upstream']['files'].values():assert hashlib.sha256((ROOT/row['local']).read_bytes()).hexdigest()==row['sha256']
    cases=[]
    for seed in [0,1,2]:
        rng=np.random.default_rng(seed);x=rng.normal(size=(12,2));y=rng.normal(size=12);query=rng.normal(size=(5,2))
        if seed==2:x[3]=x[2]
        for lam in [.01,.1,1.]:
            for eta in [.25,np.array([2.,16.])]:
                p,r,b=source_algebra(x,y,lam,eta,query);m=krr_fit(x,y,lam,eta)
                error=max(float(np.max(np.abs(p-predict_krr(m,query)))),float(np.max(np.abs(r-loo_residuals(m['alpha'],m['inverse_diagonal'])))),abs(b-m['bias']))
                assert error<1e-8
                cases.append(dict(seed=seed,regularization=lam,eta=np.asarray(eta).tolist(),max_error=error))
report={'status':'PASS','cases':cases,'max_error':max(c['max_error'] for c in cases),'scope':'18 scalar/ARD source-algebra fixtures, including duplicate inputs; pinned official source files verified','native_matlab':'NOT_RUN','optimizer_parity':'NOT_ESTABLISHED','training_reproduction':'INCOMPARABLE'}
(ROOT/'_source_check_l059_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
