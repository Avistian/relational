"""NumPy/SciPy reconstruction of the complete original TabICL checkpoint.

No teaching operator is imported. The original low-level source stages provide
comparison outputs; double precision is an algebra diagnostic, not its sklearn
wrapper's deployed float32 recipe. CLI accepts an extracted pinned wheel path.
"""
import argparse,hashlib,json,sys,types
from pathlib import Path
import numpy as np
from scipy.special import erf,softmax
from threadpoolctl import threadpool_limits
import torch
ROOT=Path(__file__).resolve().parent
SHA='f5bae1d31181a1bb4ab8e97d2a5e62a504e856f23b4ea62c7fcc2f8eec4995b6'

class NumpyTabICL:
    def __init__(self,state):self.s={k:v.detach().numpy().astype('float64') for k,v in state.items()}
    def linear(self,x,p):return x@self.s[p+'.weight'].T+self.s[p+'.bias']
    def norm(self,x,p):
        z=x-x.mean(-1,keepdims=True)
        return z/np.sqrt((z*z).mean(-1,keepdims=True)+1e-5)*self.s[p+'.weight']+self.s[p+'.bias']
    def gelu(self,x):return x*.5*(1+erf(x/np.sqrt(2)))
    def rotate(self,x):
        angles=np.arange(x.shape[-2])[:,None]*self.s['row_interactor.tf_row.rope.freqs'][None,:]
        cos=np.cos(angles);sin=np.sin(angles);out=np.empty_like(x)
        out[...,0::2]=x[...,0::2]*cos-x[...,1::2]*sin
        out[...,1::2]=x[...,0::2]*sin+x[...,1::2]*cos
        return out
    def attention(self,q,k,v,p,heads,rope=False):
        w=self.s[p+'.in_proj_weight'];b=self.s[p+'.in_proj_bias'];d=q.shape[-1];dh=d//heads
        def project(x,i):
            y=x@w[i*d:(i+1)*d].T+b[i*d:(i+1)*d]
            return np.swapaxes(y.reshape(*y.shape[:-1],heads,dh),-3,-2)
        q,k,v=project(q,0),project(k,1),project(v,2)
        if rope:q,k=self.rotate(q),self.rotate(k)
        weights=softmax((q@np.swapaxes(k,-1,-2))/np.sqrt(dh),axis=-1)
        out=np.swapaxes(weights@v,-3,-2)
        return self.linear(out.reshape(*out.shape[:-2],d),p+'.out_proj')
    def block(self,q,p,heads,k=None,rope=False):
        k=q if k is None else k
        z=self.norm(q,p+'.norm1');memory=self.norm(k,p+'.norm1')
        x=q+self.attention(z,memory,memory,p+'.attn',heads,rope)
        return x+self.linear(self.gelu(self.linear(self.norm(x,p+'.norm2'),p+'.linear1')),p+'.linear2')
    def forward(self,x,y):
        c=len(y);n,f=x.shape;trace={};features=x.T[...,None]
        z=self.linear(features,'col_embedder.in_linear')
        for i in range(3):
            p=f'col_embedder.tf_col.blocks.{i}'
            inducing=np.broadcast_to(self.s[p+'.ind_vectors'],(f,128,128))
            memory=self.block(inducing,p+'.multihead_attn1',4,z[:,:c])
            z=self.block(z,p+'.multihead_attn2',4,memory);trace[f'column_{i}']=z.copy()
        w=self.norm(self.linear(z,'col_embedder.out_w'),'col_embedder.ln_w')
        b=self.norm(self.linear(z,'col_embedder.out_b'),'col_embedder.ln_b')
        cells=np.swapaxes(features*w+b,0,1);trace['cells']=cells.copy()
        cls=np.broadcast_to(self.s['row_interactor.cls_tokens'],(n,4,128));z=np.concatenate([cls,cells],1)
        for i in range(3):
            z=self.block(z,f'row_interactor.tf_row.blocks.{i}',8,rope=True);trace[f'row_{i}']=z.copy()
        z=self.norm(z[:,:4],'row_interactor.out_ln').reshape(n,512);trace['rows']=z.copy()
        z[:c]+=self.linear(np.eye(10)[y],'icl_predictor.y_encoder')
        for i in range(12):
            z=self.block(z,f'icl_predictor.tf_icl.blocks.{i}',4,z[:c]);trace[f'icl_{i}']=z.copy()
        z=self.norm(z,'icl_predictor.ln')
        logits=self.linear(self.gelu(self.linear(z,'icl_predictor.decoder.0')),'icl_predictor.decoder.2')
        return logits[c:],trace

def original_trace(model,x,y):
    trace={};z=torch.tensor(x.T[...,None],dtype=torch.float64);c=len(y)
    with torch.no_grad():
        z=model.col_embedder.in_linear(z)
        for i,block in enumerate(model.col_embedder.tf_col.blocks):
            z=block(z,c);trace[f'column_{i}']=z.numpy().copy()
        w=model.col_embedder.ln_w(model.col_embedder.out_w(z));b=model.col_embedder.ln_b(model.col_embedder.out_b(z))
        cells=(torch.tensor(x.T[...,None])*w+b).transpose(0,1);trace['cells']=cells.numpy().copy()
        z=torch.cat([model.row_interactor.cls_tokens.expand(len(x),-1,-1),cells],1)
        for i,block in enumerate(model.row_interactor.tf_row.blocks):
            z=block(z,rope=model.row_interactor.tf_row.rope);trace[f'row_{i}']=z.numpy().copy()
        z=model.row_interactor.out_ln(z[:,:4]).flatten(-2);trace['rows']=z.numpy().copy()
        # The sklearn/inference wrapper casts labels to float32. This low-level
        # algebra fixture directly calls its unchanged label encoder in float64.
        z[:c]+=model.icl_predictor.y_encoder(torch.tensor(y,dtype=torch.float64))
        for i,block in enumerate(model.icl_predictor.tf_icl.blocks):
            z=block(z,attn_mask=c);trace[f'icl_{i}']=z.numpy().copy()
        logits=model.icl_predictor.decoder(model.icl_predictor.ln(z))
    return logits[c:].numpy(),trace

def check(source):
    manifest=json.loads((ROOT/'_sources_l066_v2.json').read_text())
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
    # Import only the unchanged model package; sklearn-wrapper dependencies are
    # irrelevant to this algebra check and are not replaced with fake solvers.
    package=types.ModuleType('tabicl');package.__path__=[str(source/'tabicl')];sys.modules['tabicl']=package
    from tabicl.model.tabicl import TabICL
    checkpoint=ROOT/'data/cache/foundation/tabicl-v1-0208.ckpt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==SHA
    c=torch.load(checkpoint,map_location='cpu',weights_only=True);model=TabICL(**c['config']);model.load_state_dict(c['state_dict'],strict=True);model.double().eval()
    # RoPE source casts positions to the frequency parameter dtype. Both paths
    # read the actual stored frequency values, including original float32 rounding.
    independent=NumpyTabICL(c['state_dict']);torch.set_num_threads(1);records=[]
    for seed,n,f,context,classes in [(7,11,3,8,2),(13,13,6,9,3),(19,10,1,6,2)]:
        rng=np.random.default_rng(seed);x=rng.normal(size=(n,f));y=np.arange(context)%classes
        logits,trace=independent.forward(x,y);reference,expected=original_trace(model,x,y)
        deltas={k:float(np.max(abs(trace[k]-expected[k]))) for k in trace};error=float(abs(logits-reference).max())
        assert max(deltas.values())<2e-9 and error<2e-9,(seed,deltas,error)
        records.append(dict(seed=seed,rows=n,features=f,context=context,classes=classes,stage_max_deltas=deltas,logit_max_delta=error))
    report=dict(status='PASS',checkpoint_sha256=SHA,tensors=len(c['state_dict']),parameter_entries=sum(v.numel() for v in c['state_dict'].values()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),cases=records,source_files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'tabicl/model').glob('*.py')},scope='Complete original model reconstructed with independent NumPy/SciPy operations: all 3 ISABs, normalized conditional affine embedding, 3 RoPE row blocks, 4 CLS outputs, label addition, all 12 context-only ICL blocks and full 10-logit head. Float64 low-level algebra on 3 fixtures; no sklearn wrapper, pretraining or benchmark claim.')
    (ROOT.parent/'reviews/lesson-quality-audit-047-070/066-independent.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):r=check(a.source)
    print(json.dumps(dict(status=r['status'],cases=len(r['cases']),max_logit_delta=max(c['logit_max_delta'] for c in r['cases']))))
