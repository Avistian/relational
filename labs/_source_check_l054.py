"""Copied-weight full forward/backward parity with pinned official numeric TabM.

Install rtdl_num_embeddings in your environment, or supply --deps DIR for a
throwaway --target install. The reference source is fetched to /tmp and hashed;
no reference implementation is used by the teaching model or its trainer.
"""
import argparse, hashlib, importlib.util, json, sys, urllib.request
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
REV='28e47ae301c92ec37787dde1ce923a0793f405b4'
URL=f'https://raw.githubusercontent.com/yandex-research/tabm/{REV}/tabm.py'

def check(deps=None):
    if deps: sys.path.insert(0, str(deps))
    path=Path('/tmp/l054-official-tabm.py')
    if not path.exists():path.write_bytes(urllib.request.urlopen(URL).read())
    source=path.read_bytes()
    expected='fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade'
    assert hashlib.sha256(source).hexdigest()==expected, 'Pinned source changed'
    spec=importlib.util.spec_from_file_location('l054_official',path)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    from relkit.tabm_v2 import TabM, member_mean_loss, ensemble_predict
    torch.set_num_threads(1)
    records=[]
    for arch in ['mini','tabm']:
      for reg in [True,False]:
        torch.manual_seed(54)
        local=TabM(4,k=3,width=7,depth=2,dropout=0.,regression=reg,arch=arch).double()
        ref=official.TabM.make(n_num_features=4,d_out=1 if reg else 2,k=3,n_blocks=2,d_block=7,dropout=0.,arch_type='tabm-mini' if arch=='mini' else 'tabm').double()
        pairs=[]
        for i,block in enumerate(local.blocks):
            rblock=ref.backbone.blocks[i][0]
            pairs.extend([(block.weight,rblock.weight,True),(block.bias,rblock.bias,False)])
            if arch=='mini':
                assert block.bias.ndim==1 and block.S is None
                assert (block.R is not None)==(i==0)
                if i==0:pairs.append((block.R,ref.backbone.affine.weight,False))
            else:
                pairs.extend([(block.R,rblock.r,False),(block.S,rblock.s,False)])
                torch.testing.assert_close(block.bias,block.bias[0].expand_as(block.bias))
                torch.testing.assert_close(block.S,torch.ones_like(block.S))
            assert block.weight.abs().max()<=1/(block.weight.shape[0]**.5)
        pairs.extend([(local.head.weight,ref.output.weight,False),(local.head.bias,ref.output.bias,False)])
        assert sum(p.numel() for p in local.parameters())==sum(p.numel() for p in ref.parameters())
        with torch.no_grad():
            for a,b,transpose in pairs:b.copy_(a.T if transpose else a)
        for independent in [False,True]:
            x=torch.randn((5,3,4) if independent else (5,4),dtype=torch.float64,requires_grad=True)
            xr=x.detach().clone().requires_grad_(True)
            y=torch.randn(5,dtype=torch.float64) if reg else torch.tensor([0,1,1,0,1])
            local.zero_grad();ref.zero_grad()
            a,b=local(x),ref(xr)
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
            la=member_mean_loss(a,y,reg)
            if reg:lb=(b[:,:,0]-y[:,None]).square().mean()
            else:lb=torch.nn.functional.cross_entropy(b.flatten(0,1),y.repeat_interleave(3))
            la.backward();lb.backward()
            torch.testing.assert_close(x.grad,xr.grad,atol=1e-12,rtol=1e-12)
            for aa,bb,transpose in pairs:torch.testing.assert_close(aa.grad.T if transpose else aa.grad,bb.grad,atol=1e-12,rtol=1e-12)
            records.append(dict(arch=arch,regression=reg,independent_batches=independent,max_output_error=float((a-b).abs().max().detach())))
    # Two common wrong TODO implementations: a scalar loss alone is not a check.
    logits=torch.tensor([[[0.,2.1972246],[0.,-2.1972246]]],requires_grad=True)
    y=torch.tensor([1]);loss=member_mean_loss(logits,y,False)
    torch.testing.assert_close(loss,torch.tensor(1.2039728))
    torch.testing.assert_close(ensemble_predict(logits,False),torch.tensor([[.5,.5]]))
    # Classification error can worsen while convex probability loss improves.
    p=torch.tensor([.51,.51,.01]); collective_error=int(p.mean()<.5)
    assert collective_error==1 and float((p<.5).float().mean())<1
    result=dict(status='PASS',official_revision=REV,official_source=URL,
                official_sha256=expected,implementation_sha256=hashlib.sha256((ROOT/'relkit/tabm_v2.py').read_bytes()).hexdigest(),cases=records,
                scope='copied weights, parameter counts, input and every parameter gradient; numeric mini/full, 2D/shared and 3D/member batches; no RNG-sequence or training parity')
    (ROOT/'_source_check_l054_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deps');check(p.parse_args().deps)
