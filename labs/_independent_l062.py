"""Parent audit: NumPy/SciPy reconstruction of full pretrained v1 computation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.special import erf,softmax
import torch
from relkit import tabpfn_l062_v2 as core
ROOT=Path(__file__).resolve().parent

def check():
    torch.set_num_threads(1)
    model,_=core.load_pretrained(core.ensure_checkpoint(ROOT));model.double()
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    def linear(x,prefix):return x@weights[prefix+'.weight'].T+weights[prefix+'.bias']
    def norm(x,prefix):
        centered=x-x.mean(-1,keepdims=True)
        return centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-5)*weights[prefix+'.weight']+weights[prefix+'.bias']
    def gelu(x):return .5*x*(1+erf(x/np.sqrt(2)))
    def numpy_forward(x,y):
        h=linear(x,'x_encoder');n=y.shape[1]
        h[:,:n]+=linear(y[...,None],'y_encoder')
        for i in range(12):
            base=f'blocks.{i}';b,t,d=h.shape
            q,k,v=linear(h,base+'.qkv').reshape(b,t,3,4,128).transpose(2,0,3,1,4)
            scores=np.einsum('bhid,bhjd->bhij',q,k[:,:,:n])/np.sqrt(128)
            mixed=np.einsum('bhij,bhjd->bhid',softmax(scores,axis=-1),v[:,:,:n])
            attention=linear(mixed.transpose(0,2,1,3).reshape(b,t,d),base+'.out')
            h=norm(h+attention,base+'.norm1')
            h=norm(h+linear(gelu(linear(h,base+'.ff1')),base+'.ff2'),base+'.norm2')
        return linear(gelu(linear(h[:,n:],'head.0')),'head.2')
    rng=np.random.default_rng(62001);x=rng.normal(size=(1,9,100));y=np.array([[0.,1.,2.,0.,1.]])
    expected=numpy_forward(x.copy(),y)
    with torch.no_grad():actual=model(torch.from_numpy(x),torch.from_numpy(y)).numpy()
    error=float(np.max(np.abs(expected-actual)));assert error<1e-10,error
    changed=x.copy();changed[:,7]*=50
    unchanged=numpy_forward(changed,y)
    other_query_error=float(np.max(np.abs(expected[:,[0,1,3]]-unchanged[:,[0,1,3]])))
    assert other_query_error==0
    permutation=np.array([4,1,3,0,2]);permuted=np.concatenate([x[:,:5][:,permutation],x[:,5:]],axis=1)
    permutation_error=float(np.max(np.abs(expected-numpy_forward(permuted,y[:,permutation]))));assert permutation_error<1e-10
    moments=[]
    for missing in [False,True]:
        z=rng.normal(size=(17,6));z[-1]*=1000
        if missing:z[2,1]=np.nan
        wanted=np.clip((z-np.nanmean(z[:10],axis=0))/(np.nanstd(z[:10],axis=0,ddof=1)+1e-6),-100,100)
        got=core.normalize_context(torch.from_numpy(z),10).numpy()
        delta=float(np.nanmax(np.abs(wanted-got)));assert delta<1e-12
        assert np.array_equal(np.isnan(wanted),np.isnan(got));moments.append(delta)
    cases=[]
    for k in [2,3,10]:
        logits=rng.normal(size=(k,4,10))*3;shifts=list(range(k))
        aligned=np.stack([logits[i,:,[(j+s)%k for j in range(k)]].T for i,s in enumerate(shifts)])
        wanted=softmax(aligned.mean(0)/.8,axis=-1)
        got=core.aggregate_views(torch.from_numpy(logits),shifts,k).numpy()
        delta=float(np.abs(wanted-got).max());assert delta<1e-12
        cases.append(dict(classes=k,delta=delta))
    result=dict(status='PASS',full_numpy_forward_max_delta=error,other_query_max_delta=other_query_error,
        context_permutation_max_delta=permutation_error,normalization_deltas=moments,ensemble_cases=cases,
        checkpoint_sha256=core.CHECKPOINT_SHA,operator_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        scope='Independent NumPy/SciPy full12layer forward with actual released weights; context-only statistics and class alignment fixtures. No new pretraining or benchmark parity.')
    (ROOT.parent/'reviews/lesson-quality-audit-047-070/062-independent.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':print(json.dumps(check(),indent=2))
