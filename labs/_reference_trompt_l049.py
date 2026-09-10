"""Copied-weight component parity against PINNED independent PyTorch Frame code.

No author-maintained Trompt repository was identified. This verifies the cell's
encoded-feature path and decoder, not original author model/optimizer parity.
"""
import ast,hashlib,json,urllib.request
from pathlib import Path
import torch
from torch import Tensor,nn
from torch.nn import GroupNorm,LayerNorm,Linear,Parameter,ReLU,Sequential
from torch.nn import functional as F
from relkit.trompt_l049 import TromptCell,TromptHead

REV='3ed76b132242be6ce5f63851052e114e57aaec5d'
SOURCES={
 'TromptConv':('conv/trompt_conv.py','e3c25aff5f0b41305ddd5095cea8166c4b3da7a3867072cbb61cdf1df0e4d684'),
 'TromptDecoder':('decoder/trompt_decoder.py','0ffabf658344dbbf358fd3306a6a8472e6fea87972648db9ac6512607d81686b')}

def check():
    torch.set_num_threads(1);ns=dict(globals(),TableConv=nn.Module,Decoder=nn.Module)
    sources=[]
    for name,(suffix,sha) in SOURCES.items():
        url=f'https://raw.githubusercontent.com/pyg-team/pytorch-frame/{REV}/torch_frame/nn/{suffix}'
        cache=Path(__file__).parent/'data/cache/l049'/Path(suffix).name
        if not cache.exists():cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(urllib.request.urlopen(url).read())
        raw=cache.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        # Execute original class body unchanged; replace only framework base class.
        tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(cache),'exec'),ns)
        sources.append(dict(url=url,sha256=sha))
    torch.manual_seed(49)
    ours=TromptCell(3,8,4).double();ref=ns['TromptConv'](8,3,4).double()
    for a,b in [('prompts','embedding_prompt'),('columns','embedding_column'),('expansion_weight','weight')]:
        with torch.no_grad():getattr(ref,b).copy_(getattr(ours,a))
    for a,b in [('prompt_norm','layer_norm_e_prompt'),('column_norm','layer_norm_e_column'),('fusion','lin'),('group_norm','group_norm')]:
        getattr(ref,b).load_state_dict(getattr(ours,a).state_dict())
    x=torch.randn(5,3,dtype=torch.double,requires_grad=True);xp=x.detach().clone().requires_grad_()
    previous=torch.randn(5,4,8,dtype=torch.double)
    a=ours(x,previous)
    encoded=ours.value_norm(F.relu(xp[:,:,None]*ours.value_weight+ours.value_bias))
    b=ref(encoded,previous)
    forward=float((a-b).abs().max().detach());a.sum().backward();b.sum().backward()
    gradient=float((x.grad-xp.grad).abs().max())
    head=TromptHead(8,2).double();decoder=ns['TromptDecoder'](8,2,4).double()
    for a,b in [(head.prompt_score,decoder.lin_attn),(head.hidden,decoder.mlp[0]),(head.norm,decoder.mlp[2]),(head.output,decoder.mlp[3])]:b.load_state_dict(a.state_dict())
    state=torch.randn(5,4,8,dtype=torch.double)
    head_gap=float((head(state)-decoder(state)).abs().max().detach())
    assert max(forward,gradient,head_gap)<1e-10
    return dict(status='PASS',cell_max_abs=forward,input_gradient_max_abs=gradient,head_max_abs=head_gap,sources=sources,
        scope='Independent PyTorch Frame cell and decoder only; numeric encoding supplied by local mirror; full model initial state differs (paper zeros vs Frame learned); no author code or optimizer parity')

if __name__=='__main__':
    r=check();Path(__file__).with_name('_reference_trompt_l049_results.json').write_text(json.dumps(r,indent=2));print(r)
