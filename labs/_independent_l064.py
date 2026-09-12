"""Independent NumPy/SciPy reconstruction of the entire historical default v2 model.

Run in the isolated TabPFN 2.0.9 source environment. Torch supplies the original
checkpoint and its fixed random group vectors; all predictive arithmetic below
uses NumPy/SciPy without importing the lesson's model implementation.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import erf,softmax
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent

def norm(x):
    centered=x-x.mean(axis=-1,keepdims=True)
    return centered/np.sqrt((centered*centered).mean(axis=-1,keepdims=True)+1e-5)

def gelu(x):return .5*x*(1+erf(x/np.sqrt(2)))

def attention(x,state,prefix,senders=None,first_head=False):
    src=x if senders is None else senders
    w=state[prefix+'._w_qkv'];out=state[prefix+'._w_out']
    q=np.einsum('...si,hdi->...hsd',x,w[0])
    k=np.einsum('...si,hdi->...hsd',src,w[1,:1] if first_head else w[1])
    v=np.einsum('...si,hdi->...hsd',src,w[2,:1] if first_head else w[2])
    # The original CPU fallback creates this scalar in the default float32 dtype.
    scores=np.matmul(q,np.swapaxes(k,-1,-2))*float(np.sqrt(np.float32(1/q.shape[-1])))
    heads=np.matmul(softmax(scores,axis=-1),v)
    return np.einsum('...hsd,hdi->...si',heads,out)

def forward(x,y,state,group_vectors):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);n=len(y)
    padded=np.pad(x,((0,0),(0,(-x.shape[-1])%2)))
    groups=padded.reshape(len(x),-1,2)
    flags=np.where(np.isnan(groups),-2.,0.)
    valid=~np.isnan(groups[:n])
    means=np.where(valid,groups[:n],0.).sum(axis=0)/np.maximum(1,valid.sum(axis=0))
    filled=np.where(np.isnan(groups),means,groups)
    mu=filled[:n].mean(axis=0);std=filled[:n].std(axis=0,ddof=1)+1e-20
    z=np.clip((filled-mu)/std,-100,100)
    active=np.maximum(1,np.any(z[1:]!=z[:1],axis=0).sum(axis=-1))
    # Integer active counts undergo default float32 division/sqrt in the source.
    z*=np.sqrt((2/active).astype(np.float32)).astype(float)[None,:,None]
    encoded=np.concatenate([z,flags],axis=-1)@state['encoder.5.layer.weight'].T
    positions=group_vectors@state['feature_positional_embedding_embeddings.weight'].T+state['feature_positional_embedding_embeddings.bias']
    encoded+=positions[None,:,:]
    yy=np.r_[y,np.repeat(y.mean(),len(x)-n)]
    ranks=(yy[:,None]>np.unique(y)).sum(axis=-1)
    yflags=np.r_[np.zeros(n),np.full(len(x)-n,-2.)]
    target=np.column_stack([ranks,yflags])@state['y_encoder.2.layer.weight'].T+state['y_encoder.2.layer.bias']
    h=np.concatenate([encoded,target[:,None,:]],axis=1)[None]
    traces=[]
    for layer in range(12):
        prefix=f'transformer_encoder.layers.{layer}'
        h=norm(h+attention(h,state,prefix+'.self_attn_between_features'))
        rows=h.transpose(0,2,1,3);context=rows[:,:,:n];query=rows[:,:,n:]
        cu=attention(context,state,prefix+'.self_attn_between_items',senders=context)
        qu=attention(query,state,prefix+'.self_attn_between_items',senders=context,first_head=True)
        h=norm(h+np.concatenate([cu,qu],axis=2).transpose(0,2,1,3))
        ff=gelu(h@state[prefix+'.mlp.linear1.weight'].T)@state[prefix+'.mlp.linear2.weight'].T
        h=norm(h+ff);traces.append(h.copy())
    final=h[0,n:,-1,:]
    head=gelu(final@state['decoder_dict.standard.0.weight'].T+state['decoder_dict.standard.0.bias'])
    logits=head@state['decoder_dict.standard.2.weight'].T+state['decoder_dict.standard.2.bias']
    return logits,traces,active

def check():
    import torch,tabpfn
    from tabpfn.model.loading import load_model
    assert tabpfn.__version__=='2.0.9';torch.set_num_threads(1)
    path=ROOT/'data/cache/foundation/tabpfn-v2.ckpt'
    model,_,_=load_model(path=path,model_seed=0);model=model.double().eval()
    state={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.state_dict().items()}
    rng=np.random.default_rng(640);cases=[]
    for name,features,classes,missing in [('binary',4,2,False),('three_classes_odd_groups',5,3,False),('missing_values',3,2,True)]:
        x=rng.normal(size=(10,features));y=np.arange(6)%classes
        if missing:x[1,0]=np.nan;x[7,1]=np.nan
        vectors=torch.randn(((features+1)//2,48),generator=torch.Generator().manual_seed(0),dtype=torch.float64).numpy()
        originals=[];handles=[]
        for layer in model.transformer_encoder.layers:
            handles.append(layer.register_forward_hook(lambda m,a,o:originals.append(o.detach().numpy().copy())))
        with torch.no_grad():reference=model(torch.from_numpy(x)[:,None,:],torch.from_numpy(y.astype(float))[:,None],single_eval_pos=6).numpy()[:,0]
        for handle in handles:handle.remove()
        with threadpool_limits(limits=1):pred,layers,active=forward(x,y,state,vectors)
        errors=[float(np.max(np.abs(a-b))) for a,b in zip(layers,originals)]
        error=float(np.max(np.abs(pred-reference)));assert error<1e-9 and max(errors)<1e-9,(name,error,max(errors))
        cases.append(dict(name=name,features=features,classes=classes,all_layer_max_abs=max(errors),logit_max_abs=error,active_channels=active.tolist()))
    report=dict(status='PASS',cases=cases,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Complete NumPy/SciPy encoding, grouped feature and asymmetric row attention, all 12 blocks and classification head against original CPU float64 source. Fixed source RNG group vectors supplied; no original wrapper or training reproduction claim.')
    (ROOT.parent/'reviews/lesson-quality-audit-047-070/064-independent.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':print(json.dumps(check(),indent=2))
