"""Pinned-source parity: outputs, token order, gradients, and optimizer update."""
import copy,hashlib,json
from pathlib import Path
import torch
from torch_frame.nn.models import ResNet
from relkit.frame_l125 import *
P=Path(__file__).resolve().parent

def check_model():
    torch.manual_seed(125);tr,q,types=typed_fixture();ds,tf=fit_and_convert(tr,q,types)
    ours=VisibleRowResNet(ds);ref=ResNet(channels=8,out_channels=8,num_layers=2,col_stats=ds.col_stats,col_names_dict=ds.tensor_frame.col_names_dict,stype_encoder_dict=encoder_recipe(),dropout_prob=0.)
    ref.load_state_dict(ours.state_dict());ours.eval();ref.eval()
    a,names=ours.tokens(tf);b,expected_names=ref.encoder(tf)
    assert names==expected_names,(names,expected_names)
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    x=ours(tf);y=ref(tf);torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
    x.square().sum().backward();y.square().sum().backward()
    errors=[]
    for (n,p),(m,q) in zip(ours.named_parameters(),ref.named_parameters()):
        assert n==m and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5);errors.append(float((p.grad-q.grad).abs().max()))
    torch.optim.Adam(ours.parameters(),lr=.001).step();torch.optim.Adam(ref.parameters(),lr=.001).step()
    torch.testing.assert_close(ours(tf),ref(tf),rtol=1e-5,atol=1e-6)
    perm=torch.tensor([1,0]);torch.testing.assert_close(ours(tf[perm]),ours(tf)[perm])
    torch.testing.assert_close(ours(tf[:1]),ours(tf)[:1],rtol=1e-5,atol=1e-6)
    return {'tokens':list(a.shape),'rows':list(x.shape),'column_order':names,'output_max_error':float((x-y).abs().max().detach()),'gradient_max_error':max(errors),'adam_update':'MATCH','permutation':'PASS','row_locality_eval':'PASS'}

if __name__=='__main__':
    files=json.loads((P/'_sources_l125.json').read_text())['files']
    for path,entry in files.items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==entry['sha256'],path
    result={'status':'PASS','pinned_files':len(files),'model':check_model()}
    (P/'_source_check_l125_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
