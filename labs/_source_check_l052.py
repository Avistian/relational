"""Execute the pinned official Model class; adapt only Faiss search and tiny utilities.
This tests neural forward/input-gradient parity, not Faiss runtime or optimizer parity.
"""
import ast,json,math,types,hashlib
from pathlib import Path
from typing import Optional,Union,Literal
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from relkit.tabr import TabRS
HERE=Path(__file__).resolve().parent

class Lambda(nn.Module):
    def __init__(self,f):super().__init__();self.f=f
    def forward(self,x):return self.f(x)

class Flat:
    def __init__(self,d):pass
    def reset(self):pass
    def add(self,x):self.x=x
    def search(self,q,n):
        v,i=((q[:,None]-self.x[None]).square().sum(-1)).topk(n,largest=False)
        return v,i

def check():
    manifest=json.loads((HERE/'_sources_l052.json').read_text())
    for f,m in manifest['files'].items():assert hashlib.sha256((HERE/'sources/l052'/f).read_bytes()).hexdigest()==m['sha256']
    source=(HERE/'sources/l052/bin__tabr.py').read_text()
    tree=ast.parse(source);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
    namespace=dict(nn=nn,torch=torch,F=F,math=math,Tensor=Tensor,Optional=Optional,Union=Union,Literal=Literal,
                   lib=types.SimpleNamespace(get_d_out=lambda n:1 if n is None or n==2 else n),
                   delu=types.SimpleNamespace(nn=types.SimpleNamespace(Lambda=Lambda)),faiss=types.SimpleNamespace(IndexFlatL2=Flat))
    exec(compile(ast.Module(body=[klass],type_ignores=[]),'<pinned TabR Model>','exec'),namespace)
    errors=[]
    for regression in [False,True]:
        torch.manual_seed(52)
        ref=namespace['Model'](n_num_features=3,n_bin_features=0,cat_cardinalities=[],n_classes=None if regression else 2,
              num_embeddings=None,d_main=8,d_multiplier=2,encoder_n_blocks=0,predictor_n_blocks=1,mixer_normalization='auto',
              context_dropout=0,dropout0=0,dropout1=0,normalization='LayerNorm',activation='ReLU')
        ours=TabRS(3,d=8,m=3,dropout=0,context_dropout=0,regression=regression)
        matched = [(ours.linear,ref.linear),(ours.K,ref.K),(ours.T,ref.T),(ours.block,ref.blocks1[0]),(ours.head,ref.head),
                     (ours.label_encoder,ref.label_encoder if regression else ref.label_encoder[0])]
        for a,b in matched:a.load_state_dict(b.state_dict())
        ref.eval();ours.eval()
        for train in [False,True]:
            x=torch.randn(9,3);y=torch.randn(9) if regression else torch.arange(9)%2
            q=x[:2].clone().requires_grad_();q2=q.detach().clone().requires_grad_()
            # Official training API receives candidates WITHOUT query batch; adds it back.
            official=ref(x_={'num':q},y=y[:2] if train else None,candidate_x_={'num':x[2:]},candidate_y=y[2:],context_size=3,is_train=train).squeeze(-1)
            cx=torch.cat([q2,x[2:]]) if train else x[2:]
            local=ours(q2,cx,y if train else y[2:],torch.arange(2) if train else None,torch.arange(9) if train else None)
            torch.testing.assert_close(local,official,atol=2e-6,rtol=2e-6)
            g1=torch.autograd.grad(official.sum(),q,retain_graph=True)[0];g2=torch.autograd.grad(local.sum(),q2,retain_graph=True)[0]
            torch.testing.assert_close(g1,g2,atol=3e-6,rtol=3e-6)
            left=[p for a,b in matched for p in a.parameters()]
            right=[p for a,b in matched for p in b.parameters()]
            local_grads=torch.autograd.grad(local.square().sum(),left)
            reference_grads=torch.autograd.grad(official.square().sum(),right)
            for ours_grad,ref_grad in zip(local_grads,reference_grads):
                torch.testing.assert_close(ours_grad,ref_grad,atol=4e-6,rtol=4e-6)
            parameter_error=max(float((a-b).abs().max()) for a,b in zip(local_grads,reference_grads))
            errors.append({'parameter_gradient_max_error':parameter_error,'parameter_tensors_checked':len(left),'regression':regression,'train':train,'logit_max_error':float((official-local).abs().max().detach()),'gradient_max_error':float((g1-g2).abs().max())})
    result={'status':'PASS','scope':'Official numeric TabR-S class; torch exact-search adapter; dropout disabled; copied weights; eval and training candidate paths; query-input and all corresponding parameter gradients','cases':errors}
    (HERE/'_source_check_l052_results.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(check())
