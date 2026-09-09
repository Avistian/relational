"""Bounded copied-weight v1-block parity and independent numerical invariances."""
import hashlib,importlib.util,json,typing
from pathlib import Path
import torch
from relkit.foundation_core import AttentionBlock,context_mask
ROOT=Path(__file__).resolve().parent


def check():
    from _check_foundation_core import checks
    from _check_benchmark_core import checks as benchmark_checks
    checks();benchmark_checks();torch.set_num_threads(1)
    manifest=json.loads((ROOT/'_sources_foundation.json').read_text())
    for file,item in manifest['reference_files'].items():
        assert hashlib.sha256((ROOT/'sources/foundation'/file).read_bytes()).hexdigest()==item['sha256'],file
    import torch.nn.modules.transformer as module
    if not hasattr(module,'Optional'):module.Optional=typing.Optional
    path=ROOT/'sources/foundation/v1-layer.py'
    spec=importlib.util.spec_from_file_location('pinned_v1_layer',path);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    cases=[]
    for context,query in [(3,2),(8,1),(2,5)]:
        torch.manual_seed(context+query)
        ref=reference.TransformerEncoderLayer(16,2,dim_feedforward=32,dropout=0.,activation='gelu',pre_norm=True).eval()
        local=AttentionBlock(16,2).eval()
        with torch.no_grad():
            local.qkv.weight.copy_(ref.self_attn.in_proj_weight);local.qkv.bias.copy_(ref.self_attn.in_proj_bias)
            local.out.load_state_dict(ref.self_attn.out_proj.state_dict())
            local.norm1.load_state_dict(ref.norm1.state_dict());local.norm2.load_state_dict(ref.norm2.state_dict())
            local.ff[0].load_state_dict(ref.linear1.state_dict());local.ff[2].load_state_dict(ref.linear2.state_dict())
        x=torch.randn(2,context+query,16,requires_grad=True);xr=x.detach().transpose(0,1).clone().requires_grad_(True)
        actual=local(x,context_mask(context,query));expected=ref(xr,src_mask=context).transpose(0,1)
        output_error=float((actual-expected).abs().max());torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        actual.square().sum().backward();expected.square().sum().backward()
        grad_error=float((x.grad-xr.grad.transpose(0,1)).abs().max());torch.testing.assert_close(x.grad,xr.grad.transpose(0,1),atol=3e-6,rtol=1e-5)
        cases.append(dict(context=context,queries=query,max_output_error=output_error,max_input_gradient_error=grad_error))
    result=dict(status='PASS',cases=cases,
        scope='Copied-weight pre-norm GELU v1 encoder block, dropout off; independent reduced-model and evaluator invariances',
        exclusions=['No full historical checkpoint architecture parity','No v2 full-layer copied-weight parity','No full TabICL block parity','No pretraining or published benchmark reproduction'],
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'relkit/foundation_core.py',ROOT/'relkit/benchmark_core.py',path]})
    (ROOT/'_source_check_foundation_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':check()
