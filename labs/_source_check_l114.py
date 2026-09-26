"""Original-source operator, gradient, BN-population and optimizer audit."""
import importlib.util,json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit.error_l114 import MLP,train_mlp
P=Path(__file__).resolve().parent

def original():
    sys.path.insert(0,str(P/'sources/l114'))
    spec=importlib.util.spec_from_file_location('original_l114',P/'sources/l114/mlp.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.MLP

def check(device='cpu'):
    torch.set_num_threads(1);torch.manual_seed(114)
    a=MLP(4,7,3,.5).to(device);b=original()(4,7,3,3,.5).to(device);b.load_state_dict(a.state_dict())
    x=torch.randn(11,4,device=device);y=torch.arange(11,device=device)%3;ids=torch.tensor([0,2,4,6,8],device=device)
    oa=torch.optim.Adam(a.parameters(),lr=.01);ob=torch.optim.Adam(b.parameters(),lr=.01)
    for m in [a,b]:m.train()
    torch.manual_seed(19);av=a(x[ids]);torch.manual_seed(19);bv=b(x[ids]);torch.testing.assert_close(av,bv,rtol=0,atol=0)
    torch.nn.functional.nll_loss(av,y[ids]).backward();torch.nn.functional.nll_loss(bv,y[ids]).backward()
    for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert ka==kb;torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
    oa.step();ob.step()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    a.eval();b.eval();torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    # Actual trainer: feature rows/labels outside train cannot alter optimization.
    xx=torch.randn(12,128);yy=torch.arange(12)%3;s={'train':torch.arange(8),'valid':torch.tensor([8,9]),'test':torch.tensor([10,11])}
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);train_mlp(xx,yy,s,4,1,p/'a',device)
        changed=xx.clone();changed[8:]=1e4;labels=yy.clone();labels[8:]=39
        train_mlp(changed,labels,s,4,1,p/'b',device)
        sa=torch.load(p/'a/checkpoint.pt',weights_only=True);sb=torch.load(p/'b/checkpoint.pt',weights_only=True)
        for k in sa:torch.testing.assert_close(sa[k],sb[k],rtol=0,atol=0)
        try:train_mlp(xx,yy,s,4,1,p/'a',device)
        except FileExistsError:pass
        else:raise AssertionError('Existing run overwritten')
    return {'status':'PASS','device':device,'original_output_gradient_adam_bn':'EXACT','heldout_features_and_labels_change_training':'NO','overwrite_rejected':True}
if __name__=='__main__':
    r=check();(P/'_source_check_l114_results.json').write_text(json.dumps(r,indent=2));print(r)
