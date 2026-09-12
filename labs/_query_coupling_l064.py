"""Reference-release query coupling through active-channel normalization.

Run in an isolated process with TabPFN 2.0.9 and scikit-learn 1.6.1.
This script executes the original implementation, not the lesson's model.
"""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import sklearn,torch,tabpfn
from tabpfn import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig

ROOT=Path(__file__).resolve().parent

def run():
    torch.set_num_threads(1)
    assert tabpfn.__version__=='2.0.9' and sklearn.__version__=='1.6.1'
    checkpoint=ROOT/'data/cache/foundation/tabpfn-v2.ckpt'
    x=np.column_stack([np.linspace(-2,2,12),np.ones(12)]).astype(np.float32)
    y=(x[:,0]>0).astype(int);queries=np.array([[.3,1],[0,2]],np.float32)
    isolated={'PREPROCESS_TRANSFORMS':[PreprocessorConfig(name='none',categorical_name='none',append_original=False,global_transformer_name=None)],
              'FEATURE_SHIFT_METHOD':None,'CLASS_SHIFT_METHOD':None,'FINGERPRINT_FEATURE':False,'OUTLIER_REMOVAL_STD':None}
    records=[]
    for name,config,views,mode,missing in [
        ('none_missing',isolated,1,'fit_preprocessors',True),
        ('none_finite_constant_control',isolated,1,'fit_preprocessors',False),
        ('none_missing_cached_control',isolated,1,'fit_with_cache',True),
        ('default_four_views_missing',None,4,'fit_preprocessors',True),
    ]:
        context=x.copy()
        if missing:context[1,1]=np.nan
        model=TabPFNClassifier(n_estimators=views,device='cpu',model_path=checkpoint,
            inference_precision=torch.float64,inference_config=config,random_state=0,n_jobs=1,fit_mode=mode).fit(context,y)
        call_counts=[]
        hook=model.model_.encoder[4].register_forward_hook(
            lambda module,args,out:call_counts.append(module.number_of_used_features_.tolist()))
        def active_count():
            count=model.model_.encoder[4].number_of_used_features_
            return None if count is None else count.tolist()
        alone=model.predict_proba(queries[:1]);alone_used=active_count()
        alone_calls=call_counts.copy();call_counts.clear()
        together=model.predict_proba(queries);together_used=active_count()
        together_calls=call_counts.copy();hook.remove()
        delta=float(np.abs(alone[0]-together[0]).max())
        if name=='none_missing':assert delta>1e-3 and alone_used==[[1]] and together_used==[[2]]
        if name.endswith('control'):assert delta<2e-7
        records.append(dict(name=name,views=views,fit_mode=mode,missing_context_value=missing,
            alone=alone.tolist(),together=together.tolist(),existing_query_max_abs_change=delta,
            active_main_channels_alone=alone_used,active_main_channels_together=together_used,
            per_view_active_channels_alone=alone_calls,per_view_active_channels_together=together_calls))
    report=dict(status='PASS',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,tabpfn=tabpfn.__version__),
        context_before_missing=x.tolist(),missing_location=[1,1],labels=y.tolist(),queries=queries.tolist(),records=records,
        precision='CPU float64 model inference; original public classifier returns float32 probabilities',
        mechanism='Missing context values keep an otherwise constant column in wrapper selection; mean imputation makes its main channel constant. An appended query can activate it, changing all-row active-channel normalization before attention.',
        scope='Measured reference release behavior with explicit one-view configuration and separately tested default/cached controls. Attention remains context-only; no assertion of universal coupling across configurations.')
    return report

if __name__=='__main__':
    result=run();(ROOT/'_query_coupling_l064_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
