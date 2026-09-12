"""A source-confirmed query-batch counterexample in historical power preprocessing."""
import hashlib,importlib.metadata,json,warnings
from pathlib import Path
import torch
from sklearn.preprocessing import PowerTransformer
from relkit import tabpfn_l062_v2 as core
from _check_l062_v2 import official_source
ROOT=Path(__file__).resolve().parent
CONTEXT=[4.035009215641594e-9,1.1499681207283174e-9,2.4437400991672575e-9,
    2.4904684980953107e-9,2.927720288425917e-9,3.7869735125539705e-10,
    4.926330454679828e-10,4.151606558533416e-10,2.2727798809629718e-10,
    3.3902582963207806e-9,4.9853734473970235e-9,6.020752230107007e-10]

def check():
    torch.set_num_threads(1);x=torch.tensor(CONTEXT)[:,None];q=x[:1].clone();extra=torch.tensor([[-100.]])
    y=torch.arange(12)%2;both=torch.cat([q,extra]);path=core.ensure_checkpoint(ROOT)
    model,_=core.load_pretrained(path)
    a,_=core.predict_numeric(model,x,y,q,views=4)
    b,_=core.predict_numeric(model,x,y,both,views=4)
    # Negative control: the one-view configuration uses no power transform.
    plain_a,_=core.predict_numeric(model,x,y,q,views=1)
    plain_b,_=core.predict_numeric(model,x,y,both,views=1)
    assert (plain_a-plain_b[:1]).abs().max()<1e-5
    z=core.normalize_context(torch.cat([x,both]),12).numpy()
    pt=PowerTransformer().fit(z[:12]);exception=None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:pt.transform(z)
        except RuntimeWarning as error:exception=str(error)
    assert exception and 'overflow' in exception
    transformed_alone=core.preprocess_numeric(torch.cat([x,q]),12,'power_all')[0]
    transformed_batch=core.preprocess_numeric(torch.cat([x,both]),12,'power_all')[0]
    loader,predict,_,_=official_source();reference,_=loader(str(path.parent),path.name,'cpu');reference=reference[2]
    def original(query):
        return predict(reference,torch.cat([x,query])[:,None],torch.cat([y,torch.zeros(len(query))])[:,None],12,
            inference_mode=True,preprocess_transform='mix',N_ensemble_configurations=4,feature_shift_decoder=True,seed=0)[0]
    oa=original(q);ob=original(both)
    parity=max(float((a-oa).abs().max()),float((b-ob).abs().max()));assert parity<1e-5
    delta=float((a-b[:1]).abs().max());assert delta>.05
    report=dict(status='PASS',context=CONTEXT,labels=y.tolist(),query=q.tolist(),extra_query=extra.tolist(),
        power_lambda=pt.lambdas_.tolist(),transform_exception=exception,
        preprocessing_max_delta=float((transformed_alone-transformed_batch[:-1]).abs().max()),
        local_alone=a.tolist(),local_batched=b.tolist(),official_alone=oa.tolist(),official_batched=ob.tolist(),
        prediction_max_delta=delta,source_parity_max_delta=parity,none_view_max_delta=float((plain_a-plain_b[:1]).abs().max()),
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn']},
        operator_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),checkpoint_sha256=core.CHECKPOINT_SHA,
        finding='Context-only fitting does not guarantee query-batch independence when query transformation failure triggers whole-column fallback. Both historical source and visible wrapper exhibit this counterexample in the recorded environment.')
    (ROOT/'_query_fallback_l062_results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':print(json.dumps(check(),indent=2))
