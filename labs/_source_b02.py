"""Independent scalar oracle for official packed linear and cosine embedding layers."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];P=R/'labs';S=P/'sources/b02/upstream';E=P/'evidence/b02'
sys.path.insert(0,str(S/'src'))
from project.nn import LinearPack,CosineEmbeddings
from relkit.embeddings_b02 import member_predictions

def audit():
    torch.set_num_threads(1);torch.manual_seed(202)
    layer=LinearPack([2,3],[2,1],pack_size=2,max_in_features=3,max_out_features=2).double()
    raw=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.float64)[:,None,:]
    x=raw*mask
    result=layer(x)
    expected=torch.zeros_like(result)
    for k,(ni,no) in enumerate([(2,2),(3,1)]):
        for b in range(4):
            for o in range(no):expected[k,b,o]=sum(x[k,b,i]*layer.weight[k,o,i] for i in range(ni))+layer.bias[k,o]
    torch.testing.assert_close(result,expected,atol=1e-12,rtol=1e-12)
    g=torch.autograd.grad(result.square().sum(),(raw,layer.weight,layer.bias),retain_graph=True)
    h=torch.autograd.grad(expected.square().sum(),(raw,layer.weight,layer.bias),retain_graph=True)
    for a,b in zip(g,h):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    emb=CosineEmbeddings(2,4,init_scale=.2).double();inputs=torch.tensor([[.3,-.4],[.9,1.2]],dtype=torch.float64,requires_grad=True)
    assert torch.count_nonzero(emb(inputs))==0
    with torch.no_grad():
        emb.elementwise_affine_weight.fill_(1.2);emb.elementwise_affine_bias.fill_(.1);emb.bias.fill_(.25)
    z=emb(inputs)
    explicit=1.2*torch.cat([inputs[...,None],torch.cos(2*torch.pi*(inputs[...,None]*emb.weight+emb.bias))],dim=-1)+.1
    torch.testing.assert_close(z,explicit,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(z.sum(),inputs,retain_graph=True)[0];b=torch.autograd.grad(explicit.sum(),inputs)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    # Cross-member gradients: one pack member cannot update another member's weights.
    isolated=layer(x)[0].sum();grad=torch.autograd.grad(isolated,layer.weight)[0]
    assert torch.count_nonzero(grad[1])==0
    r=dict(status='PASS',packed_forward_max_error=float((result-expected).abs().max().detach()),packed_all_gradients='PASS',cross_member_weight_gradient='ZERO',cosine_forward_input_gradient='PASS',zero_initial_embedding='PASS',source_formula='alpha * concat(x, cos(2*pi*(w*x+b))) + beta',scope='Copied operator arithmetic, not whole-training parity',torch_version=torch.__version__)
    (E/'mechanism-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':audit()
