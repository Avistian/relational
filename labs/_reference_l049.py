"""Validate the numeric released forward path with transplanted weights.

The reference is fetched only for validation, pinned by commit and SHA256.
lib is adapted only for the released tanglu activation; the entire inspected
Tokenizer/MultiheadAttention/ExcelFormer class source executes unchanged.
"""
import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path
import torch
from relkit.claim_models import ExcelFormer

REVISION = "17f70526390e70390bb8c8ec3850697eb730f9cd"
URL = f"https://raw.githubusercontent.com/WhatAShot/ExcelFormer/{REVISION}/bin/excel_former.py"
SHA = "f3b0e8fb7ba5968f956bc042adfb8293784bee06b70cc51e88cb288b01ce470c"

def validate(path=None):
    torch.set_num_threads(1)
    path = Path(path) if path else Path(__file__).parent/'data/cache/l049/excel_former.py'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL,path)
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    shim=types.ModuleType('lib')
    shim.get_activation_fn=lambda name: lambda x: x.chunk(2,-1)[0]*torch.tanh(x.chunk(2,-1)[1])
    old=sys.modules.get('lib');sys.modules['lib']=shim
    ns={'__name__':'l049_released_reference'}
    try: exec(compile(raw,str(path),'exec'),ns)
    finally:
        if old is None: sys.modules.pop('lib',None)
        else: sys.modules['lib']=old
    torch.manual_seed(49)
    ours=ExcelFormer(5,d=16,heads=2,layers=2,dropout=0).double().eval()
    ref=ns['ExcelFormer'](d_numerical=5,categories=None,token_bias=True,n_layers=2,
        d_token=16,n_heads=2,attention_dropout=0,ffn_dropout=0,residual_dropout=0,
        prenormalization=True,kv_compression=None,kv_compression_sharing=None,d_out=1,init_scale=.01).double().eval()
    ref.tokenizer.load_state_dict(ours.tokenizer.state_dict())
    for a,b in zip(ours.blocks,ref.layers):
        for src,dst in [('q','W_q'),('k','W_k'),('v','W_v'),('out','W_out')]:
            getattr(b['attention'],dst).load_state_dict(getattr(a,src).state_dict())
        b['linear0'].load_state_dict(a.gate.state_dict())
        b['norm1'].load_state_dict(a.norm1.state_dict())
        if 'norm0' in b:b['norm0'].load_state_dict(a.norm0.state_dict())
    for a,b in [('pool','last_fc'),('norm','last_normalization'),('act','last_activation'),('head','head')]:
        getattr(ref,b).load_state_dict(getattr(ours,a).state_dict())
    x=torch.randn(7,5,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    a,b=ours(x),ref(y,None)
    gap=float((a-b).abs().max().detach())
    a.sum().backward();b.sum().backward()
    gradient=float((x.grad-y.grad).abs().max())
    assert gap<1e-10 and gradient<1e-10,(gap,gradient)
    return {'status':'PASS','forward_max_abs':gap,'input_gradient_max_abs':gradient,
            'source_url':URL,'source_sha256':SHA,'scope':'numeric/pre-norm/no-compression/eval forward, copied weights; no optimizer parity'}

if __name__=='__main__':
    result=validate(sys.argv[1] if len(sys.argv)>1 else None)
    Path(__file__).with_name('_reference_l049_results.json').write_text(json.dumps(result,indent=2))
    print(result)
