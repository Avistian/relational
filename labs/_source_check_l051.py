"""Execute only audited upstream functions, without importing its training stack."""
import ast,contextlib,io,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal,special_ortho_group
from sklearn.covariance import EmpiricalCovariance,MinCovDet
from relkit.bias_interventions import smooth_targets,rotate_splits
root=Path(__file__).parent
text=(root/'sources/l051/data_transforms.py').read_text()
names={'remove_high_frequency_from_train','apply_random_rotation'}
nodes=[n for n in ast.parse(text).body if getattr(n,'name',None) in names]
ns=dict(np=np,multivariate_normal=multivariate_normal,special_ortho_group=special_ortho_group,
        EmpiricalCovariance=EmpiricalCovariance,MinCovDet=MinCovDet)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned upstream functions>','exec'),ns)
rng=np.random.RandomState(7);x=rng.normal(size=(30,3));v=rng.normal(size=(10,3));y=(x[:,0]>.2).astype(int);yv=np.arange(10)%2
with contextlib.redirect_stdout(io.StringIO()):
    reference=ns['remove_high_frequency_from_train'](x,v,v,y,yv,yv,cov_mult=.49,covariance_estimation='classic')
    _,labels=smooth_targets(x,y,.7,EmpiricalCovariance().fit(x).covariance_)
    soft=ns['remove_high_frequency_from_train'](x,v,v,y,yv,yv,cov_mult=.49,covariance_estimation='classic',classif=False)[3]
    probability,_=smooth_targets(x,y,.7,EmpiricalCovariance().fit(x).covariance_)
assert np.array_equal(labels,reference[3]);assert np.allclose(probability,soft,atol=1e-12)
assert np.array_equal(reference[4],yv) and np.array_equal(reference[5],yv)
ref=ns['apply_random_rotation'](x,v,v,y,yv,yv,rng=np.random.RandomState(9))
q=special_ortho_group.rvs(3,random_state=np.random.RandomState(9))
assert all(np.allclose(a,b) for a,b in zip(ref[:3],rotate_splits([x,v,v],q)))
result=dict(smoothing_max_error=float(np.max(np.abs(probability-soft))),hard_labels_equal=True,
            rotation_equal=True,heldout_labels_unchanged=True,
            scope='Classic nonsingular covariance, h=.7, upstream cov_mult=.49; robust estimator and complete trainer parity not asserted',
            sha256=hashlib.sha256(text.encode()).hexdigest())
(root/'_reference_l051_results.json').write_text(json.dumps(result,indent=2));print(result)
