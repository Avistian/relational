"""Historical TabICL power-transform fallback couples query preprocessing.

Run with the pinned reference dependencies prepared by _check_l066_v2.py.
This checks a one-power-view release extension, separate from the lesson's
one-view normalization='none' experiment and from the full paper ensemble.
"""
import hashlib,importlib.metadata,json,sys,warnings
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sources/l066-v2'))
from tabicl import TabICLClassifier
warnings.filterwarnings('ignore',category=RuntimeWarning)

def check():
    torch.set_num_threads(1)
    checkpoint=ROOT/'data/cache/foundation/tabicl-v1-0208.ckpt'
    expected='f5bae1d31181a1bb4ab8e97d2a5e62a504e856f23b4ea62c7fcc2f8eec4995b6'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    rng=np.random.default_rng(0);x=rng.exponential(size=(30,1))*1e-10;y=(x[:,0]>np.median(x)).astype(int)
    query=np.array([[1e-7]]);expanded=np.concatenate([query,[[1.],[-1.]]]);records=[]
    for method in ['none','power']:
        classifier=TabICLClassifier(n_estimators=1,norm_methods=method,model_path=checkpoint,checkpoint_version='tabicl-classifier-v1-0208.ckpt',allow_auto_download=False,device='cpu',n_jobs=1,use_amp=False,inference_config={k:dict(offload=False) for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
        classifier.fit(x,y)
        before=classifier.predict_proba(query)[0];after=classifier.predict_proba(expanded)[0]
        pre=classifier.ensemble_generator_.preprocessors_[method]
        z_before=pre.transform(query)[0];z_after=pre.transform(expanded)[0]
        failed=False;error=None
        if method=='power':
            try:pre.normalizer_.transform(pre.standard_scaler_.transform(expanded))
            except ValueError as e:failed=True;error=str(e)
        records.append(dict(normalization=method,probability_before=before.tolist(),probability_after=after.tolist(),probability_max_delta=float(abs(before-after).max()),encoded_before=z_before.tolist(),encoded_after=z_after.tolist(),unclipped_power_transform_raises=failed,error=error,lambdas=pre.normalizer_.lambdas_.tolist() if method=='power' else None))
    assert records[0]['probability_max_delta']<2e-5 and records[0]['encoded_before']==records[0]['encoded_after']
    assert records[1]['unclipped_power_transform_raises'] and records[1]['encoded_before']!=records[1]['encoded_after']
    assert records[1]['probability_max_delta']>1e-5,records
    r=dict(status='PASS',checkpoint_sha256=expected,context=x.tolist(),context_labels=y.tolist(),existing_query=query.tolist(),expanded_queries=expanded.tolist(),records=records,versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn']},source_sha256=hashlib.sha256((ROOT/'sources/l066-v2/tabicl/sklearn/preprocessing.py').read_bytes()).hexdigest(),scope='Actual original v1 checkpoint with historical 0.1.4 sklearn wrapper. Fixed context/model/class order; only append unlabeled queries. One explicit power view triggers ValueError then clips the complete query batch to training extrema before retry. The one-none-view control remains unchanged. This is a preprocessing effect, not query-query attention, and not a full 32-view paper reproduction.')
    (ROOT/'_query_coupling_l066_results.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':print(json.dumps(check(),indent=2))
