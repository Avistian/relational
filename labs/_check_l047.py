"""Meaningful mechanism checks; optional weight-transplant against pinned official source.
Run: python labs/_check_l047.py [--reference /path/to/official/models/model.py]
Reference check needs einops; it is only a validation dependency.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relkit.saint import (pack_rows, unpack_rows, attention_weights, info_nce,
                         SAINT, SaintStage, predict_saint, train_saint)
from relkit.saint_experiment import prepare


def check(reference=None):
    torch.set_num_threads(1)
    torch.manual_seed(47)
    x = torch.randn(5, 4, 8, dtype=torch.float64)
    assert torch.equal(unpack_rows(pack_rows(x), 4), x)
    assert pack_rows(x).shape == (1, 5, 32)
    q, k, v = [torch.randn(2, 3, 5, 7, dtype=torch.float64) for _ in range(3)]
    a = attention_weights(q, k)
    err = float((a @ v - F.scaled_dot_product_attention(q,k,v)).abs().max())
    assert err < 1e-12 and torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
    z = torch.eye(4)
    assert info_nce(z,z) < info_nce(z,z.roll(1,0))
    assert abs(float(info_nce(torch.zeros(4,3),torch.zeros(4,3))) - np.log(4)) < 1e-6
    row = SaintStage(4,8,ff_dropout=0.).double().eval()
    col = SaintStage(4,8,ff_dropout=0.,variant='col').double().eval()
    changed = x.clone(); changed[1] = torch.randn_like(changed[1]) * 3
    delta = float((row(x)[0]-row(changed)[0]).abs().max().detach())
    assert delta > 1e-7
    assert torch.allclose(col(x)[0],col(changed)[0],atol=1e-12)
    perm = torch.tensor([3,1,4,0,2])
    assert torch.allclose(row(x)[perm],row(x[perm]),atol=1e-10)
    one = attention_weights(q[:,:,:1], k[:,:,:1])
    assert torch.equal(one,torch.ones_like(one))
    fr = prepare('diabetes')
    assert set(fr['train']).isdisjoint(fr['test']) and set(fr['valid']).isdisjoint(fr['test'])
    # Smoke a real training pass and require exact reproducibility with seed BEFORE construction.
    predictions = []
    for _ in range(2):
        torch.manual_seed(47)
        model = SAINT(fr['xn'].shape[1],fr['cards'],d=4,heads=2)
        model, _ = train_saint(model,fr['xn'],fr['xc'],fr['y'],fr['train'],fr['valid'],seed=47,epochs=1)
        predictions.append(predict_saint(model,fr['xn'][fr['test']],fr['xc'][fr['test']]))
    assert np.array_equal(*predictions)
    out = {'attention_reference_max_error': err, 'companion_change_max_abs': delta,
           'row_permutation_equivariance': 'PASS', 'feature_only_companion_invariance': 'PASS',
           'identical_seed_training': 'PASS', 'info_nce_pairing': 'PASS', 'reference_block': 'NOT_RUN'}
    if reference:
        spec = importlib.util.spec_from_file_location('official_saint',reference)
        ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
        official = ref.RowColTransformer(num_tokens=1, dim=8, nfeats=4, depth=1,
                                        heads=4, dim_head=16, attn_dropout=0., ff_dropout=0., style='colrow').double().eval()
        ours = SaintStage(4,8,heads=4,ff_dropout=0.).double().eval()
        for dst, src in zip(list(ours.col)+list(ours.row), official.layers[0]):
            dst.norm.load_state_dict(src.norm.state_dict())
            source_fn = src.fn.fn
            dst.fn.load_state_dict(source_fn.state_dict() if hasattr(source_fn,'to_qkv') else source_fn.net.state_dict())
        x1 = x.clone().requires_grad_(); x2 = x.clone().requires_grad_()
        y1, y2 = ours(x1), official(x2)
        y1.square().sum().backward(); y2.square().sum().backward()
        e = float((y1-y2).abs().max().detach()); g = float((x1.grad-x2.grad).abs().max())
        assert e < 1e-10 and g < 1e-9
        out.update(reference_block='PASS', reference_forward_max_error=e,
                   reference_gradient_max_error=g,
                   reference_source_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest())
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--reference'); args=p.parse_args()
    result=check(args.reference)
    Path(__file__).with_name('_check_l047_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
