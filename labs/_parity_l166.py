"""Independent released model oracle: checkpoint outputs, gradients, masking, batching."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from relkit import rdbpfn_l166 as course
P=Path(__file__).resolve().parent
assert hasattr(course,'RDBPFN'),'Visible RDBPFN implementation missing'
spec=importlib.util.spec_from_file_location('l166_original',P/'sources/l166/upstream/model_pretrain/src/models.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
torch.set_num_threads(1);torch.manual_seed(42)
results=[]
for arm,filename in [('RDBPFN','model_eval00528.pt'),('RDBPFN_single','model_eval00360.pt')]:
    ref=original.build_model(original.ModelConfig(num_layers=6)).eval()
    original.load_checkpoint(ref,P/'evidence/l166/checkpoints'/(arm+'.pt'),'cpu')
    model=course.RDBPFN().eval();model.load_state_dict(ref.state_dict(),strict=True)
    ref.double();model.double()
    x=torch.randn(1,7,3,dtype=torch.float64);y=torch.tensor([[0.,1.,0.,1.]],dtype=torch.float64)
    a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    r=ref((a,y),4);s=model((b,y),4)
    torch.testing.assert_close(r,s,atol=1e-10,rtol=1e-10)
    r.sum().backward();s.sum().backward();torch.testing.assert_close(a.grad,b.grad,atol=1e-9,rtol=1e-9)
    altered=x.clone();altered[:,5:]=1000
    torch.testing.assert_close(model((x,y),4)[:,0],model((altered,y),4)[:,0],atol=1e-10,rtol=1e-10)
    single=model((x[:,:5],y),4)
    torch.testing.assert_close(single[:,0],s[:,0],atol=1e-10,rtol=1e-10)
    results.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),max_output_delta=float((r-s).abs().max().detach()),max_gradient_delta=float((a.grad-b.grad).abs().max()),query_independence='PASS',chunk_invariance='PASS'))
r=dict(status='PASS',checkpoint_parity=results)
(P/'evidence/l166/parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
