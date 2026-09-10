"""Pinned supervised-path parity; source imports validate, never train the lab.

The vocabulary is deliberately remapped to the local missing/unseen convention.
This is copied-weight function parity, not initialization or training parity.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from relkit.saint import SAINT

SOURCE = Path(__file__).parent / 'sources/l047'


def verify_sources():
    manifest = json.loads((SOURCE / 'provenance.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
    return manifest


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_supervised_path():
    manifest = verify_sources()
    package = types.ModuleType('_l047_official')
    package.__path__ = [str(SOURCE / 'models')]
    sys.modules[package.__name__] = package
    load_source('_l047_official.model', SOURCE / 'models/model.py')
    ref = load_source('_l047_official.pretrainmodel', SOURCE / 'models/pretrainmodel.py')
    aug = load_source('_l047_augmentations', SOURCE / 'augmentations.py')
    cases = []
    # Include empty feature families, missing values, all three variants, and recurrence.
    for n_num, cards in [(2, [3, 4]), (2, []), (0, [3, 4])]:
        for variant in ['col', 'row', 'colrow']:
            torch.manual_seed(47)
            official = ref.SAINT(categories=tuple([1] + cards), num_continuous=n_num,
                                 dim=4, depth=2, heads=2, attentiontype=variant,
                                 final_mlp_style='sep').eval()
            ours = SAINT(n_num, cards, d=4, depth=2, heads=2,
                         ff_dropout=0., variant=variant).eval()
            with torch.no_grad():
                ours.cls.copy_(official.embeds.weight[:1].unsqueeze(0))
                for j, emb in enumerate(ours.cats):
                    offset = int(official.categories_offset[j + 1])
                    emb.weight.copy_(official.embeds.weight[offset:offset + cards[j]])
                    # Local code 0 means missing OR unseen. Official missingness uses
                    # a separate mask table; adapt its weights, not its vocabulary fit.
                    emb.weight[0].copy_(official.mask_embeds_cat.weight[2 * (j + 1)])
                for j, mlp in enumerate(ours.nums):
                    mlp.load_state_dict(official.simple_MLP[j].layers.state_dict())
                    ours.missing_num[j].copy_(official.mask_embeds_cont.weight[2 * j])
                ours.head.load_state_dict(official.mlpfory.layers.state_dict())
                for stage, layers in zip(ours.stages, official.transformer.layers):
                    for dst, src in zip(list(stage.col) + list(stage.row), layers):
                        dst.norm.load_state_dict(src.norm.state_dict())
                        fn = src.fn.fn
                        dst.fn.load_state_dict(fn.state_dict() if hasattr(fn, 'to_qkv') else fn.net.state_dict())
            xn = torch.randn(3, n_num)
            if n_num:
                xn[1, 0] = float('nan')
            xn.requires_grad_()
            ref_num = xn.detach().nan_to_num().requires_grad_()
            xc = torch.stack([torch.tensor([0, 1, c - 1]) for c in cards], 1) if cards else torch.empty(3, 0, dtype=torch.long)
            ref_cat = torch.cat([torch.zeros(3, 1, dtype=torch.long), xc], 1)
            cat_mask = torch.cat([torch.ones(3, 1, dtype=torch.long), (xc != 0).long()], 1)
            _, ec, en = aug.embed_data_mask(ref_cat, ref_num, cat_mask, (~xn.isnan()).long(), official)
            tokens = torch.cat([ec, en], 1)
            y_ref = official.mlpfory(official.transformer(ec, en)[:, 0])
            y_ours = ours(xn, xc)
            token_error = float((ours.tokenize(xn, xc) - tokens).abs().max().detach())
            output_error = float((y_ref - y_ours).abs().max().detach())
            y_ref.square().sum().backward()
            y_ours.square().sum().backward()
            input_error = float((xn.grad - ref_num.grad).abs().max()) if n_num else 0.
            head_error = float((ours.head[0].weight.grad - official.mlpfory.layers[0].weight.grad).abs().max())
            assert max(token_error, output_error, input_error, head_error) < 2e-5
            cases.append(dict(n_num=n_num, cards=cards, variant=variant, depth=2,
                              token_max_error=token_error, logits_max_error=output_error,
                              numeric_gradient_max_error=input_error, head_gradient_max_error=head_error))
    return dict(status='PASS', dtype='float32', cases=cases, source_commit=manifest['commit'],
                scope='Copied weights and explicitly remapped missing/unseen category token; no optimizer, initialization, vocabulary-fit, or score parity.')
