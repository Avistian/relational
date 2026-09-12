"""Independent NumPy/SciPy arithmetic for the complete released temporal PFN.

Original source supplies only state tensors, fixed random group vectors and
comparison outputs. This module does not import the teaching implementation.
"""
import numpy as np
from scipy.special import erf, softmax


def norm(x):
    centered = x-x.mean(axis=-1, keepdims=True)
    return centered/np.sqrt(np.mean(centered*centered, axis=-1, keepdims=True)+1e-5)


def gelu(x):
    return .5*x*(1+erf(x/np.sqrt(2)))


def attention(x, state, prefix, senders=None, first_head=False):
    source = x if senders is None else senders
    weight = state[prefix+'._w_qkv']; out = state[prefix+'._w_out']
    q = np.einsum('...si,hdi->...hsd', x, weight[0])
    k = np.einsum('...si,hdi->...hsd', source, weight[1, :1] if first_head else weight[1])
    v = np.einsum('...si,hdi->...hsd', source, weight[2, :1] if first_head else weight[2])
    # The pinned release takes Torch >= 2 SDPA, not the older rounded fallback.
    scores = np.matmul(q, np.swapaxes(k, -1, -2))/np.sqrt(q.shape[-1])
    heads = np.matmul(softmax(scores, axis=-1), v)
    return np.einsum('...hsd,hdi->...si', heads, out)


def forward(x, y, times, state, group_vectors):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    context = len(y); rows = len(x)
    groups = np.pad(x, ((0, 0), (0, (-x.shape[1]) % 2))).reshape(rows, -1, 2)
    # The release selects using every supplied row and packs within each group.
    packed = np.zeros_like(groups)
    for g in range(groups.shape[1]):
        keep = np.any(groups[1:, g] != groups[:1, g], axis=0)
        packed[:, g, :int(keep.sum())] = groups[:, g, keep]
    finite = np.isfinite(packed[:context])
    means = np.where(finite, packed[:context], 0.).sum(axis=0)/(finite.sum(axis=0)+1e-10)
    filled = np.where(np.isfinite(packed), packed, means)
    mu = filled[:context].mean(axis=0)
    std = np.sqrt(np.sum((filled[:context]-mu)**2, axis=0)/(context-1))+1e-6
    if context == 1:std = np.ones_like(std)
    z = np.clip((filled-mu)/std, -100, 100)
    active = np.maximum(1, np.any(z[1:] != z[:1], axis=0).sum(axis=-1))
    z *= np.sqrt((2/active).astype(np.float32)).astype(float)[None, :, None]
    key = next(k for k in state if k.startswith('encoder.') and k.endswith('.layer.weight'))
    numeric_width = state[key].shape[1]
    time_representation = None
    if numeric_width > 2:
        times = np.asarray(times, dtype=np.float64).reshape(rows)
        low, high = times[:context].min(), times[:context].max(); span = high-low
        normalized = np.clip((times-low)/(span+float(span < 1e-16)), -5, 6)
        temporal_keys = [k for k in state if k.endswith('linear_time_transform.weight')]
        if temporal_keys:
            temporal = temporal_keys[0]
            time_representation = normalized[:, None]@state[temporal].T+state[temporal.replace('weight', 'bias')]
            time_representation[:, 1:] = np.sin(time_representation[:, 1:])
        else:time_representation = normalized[:, None]
        z = np.concatenate([z, np.repeat(time_representation[:, None], z.shape[1], axis=1)], axis=-1)
    encoded = z@state[key].T+state[key.replace('weight', 'bias')]
    positions = group_vectors@state['feature_positional_embedding_embeddings.weight'].T+state['feature_positional_embedding_embeddings.bias']
    encoded += positions[None]
    yy = np.r_[y, np.full(rows-context, y.mean())]
    ranks = (yy[:, None] > np.unique(y)).sum(axis=-1)
    flags = np.r_[np.zeros(context), np.full(rows-context, -2.)]
    target = np.column_stack([ranks, flags])@state['y_encoder.2.layer.weight'].T+state['y_encoder.2.layer.bias']
    h = np.concatenate([encoded, target[:, None]], axis=1)[None]
    initial = h.copy(); traces = []
    for layer in range(12):
        prefix = f'transformer_encoder.layers.{layer}'
        h = norm(h+attention(h, state, prefix+'.self_attn_between_features'))
        by_group = h.transpose(0, 2, 1, 3); c = by_group[:, :, :context]; q = by_group[:, :, context:]
        ca = attention(c, state, prefix+'.self_attn_between_items', senders=c)
        qa = attention(q, state, prefix+'.self_attn_between_items', senders=c, first_head=True)
        h = norm(h+np.concatenate([ca, qa], axis=2).transpose(0, 2, 1, 3))
        h = norm(h+gelu(h@state[prefix+'.mlp.linear1.weight'].T)@state[prefix+'.mlp.linear2.weight'].T)
        traces.append(h.copy())
    final = h[0, context:, -1]
    logits = gelu(final@state['decoder_dict.standard.0.weight'].T+state['decoder_dict.standard.0.bias'])@state['decoder_dict.standard.2.weight'].T+state['decoder_dict.standard.2.bias']
    return logits, traces, dict(initial=initial, active=active, time_representation=time_representation)


def check():
    import contextlib, hashlib, io, json, sys, types
    from pathlib import Path
    import torch
    from threadpoolctl import threadpool_limits
    from _sources_l068_v2 import ensure_source
    root = Path(__file__).resolve().parent
    source = ensure_source()
    package = types.ModuleType('tabpfn'); package.__path__ = [str(source/'tabpfn')]
    sys.modules['tabpfn'] = package
    from tabpfn.scripts.model_builder import load_model
    torch.set_num_threads(1)
    cases = []
    for family in ['base', 'dist', 'dist_ablation_no_t2v']:
        path = root/f'data/cache/l068-release/tabpfn/model_cache/tabpfn_{family}_model_1.cpkt'
        with contextlib.redirect_stdout(io.StringIO()):
            loaded, config = load_model(str(path), 'cpu', verbose=False)
        model = loaded[2].double().eval()
        state = {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        for name, features in [('one_feature', 1), ('constant_and_missing', 3), ('six_features', 6), ('constant_source_time', 3)]:
            rng = np.random.default_rng(680)
            x = rng.normal(size=(11, features)); x[2, 0] = np.nan
            if features > 1:x[:, 1] = 3.
            y = np.array([0.,1,2,0,1,2])
            c = np.array([0.,0,1,1,2,2,3,3,4,5,30])
            if name == 'constant_source_time':c[:6] = 2.
            model.generator_device = torch.device('cpu'); model.generator.manual_seed(17)
            vectors = torch.randn(((features+1)//2, 48), generator=torch.Generator().manual_seed(17), dtype=torch.float64).numpy()
            original = {}; hooks = []
            def capture(key):
                return lambda m,a,v:original.update({key:v.detach().numpy().copy()})
            for i, layer in enumerate(model.transformer_encoder.layers):
                hooks.append(layer.register_forward_hook(capture(str(i))))
            hooks.append(model.transformer_encoder.register_forward_pre_hook(lambda m,a:original.update(initial=a[0].detach().numpy().copy())))
            inputs = {'main':torch.from_numpy(x.copy())[:,None]}
            if family != 'base':inputs['dist_shift_domain'] = torch.from_numpy(c.copy())[:,None,None]
            with torch.no_grad():
                reference = model((inputs, torch.from_numpy(y.copy())[:,None]), single_eval_pos=6).numpy()[:,0]
            for hook in hooks:hook.remove()
            with threadpool_limits(limits=1):
                logits, layers, extras = forward(x,y,c,state,vectors)
            errors = {str(i):float(np.max(np.abs(layers[i]-original[str(i)]))) for i in range(12)}
            errors['input'] = float(np.max(np.abs(extras['initial']-original['initial'])))
            errors['logits'] = float(np.max(np.abs(logits-reference)))
            assert max(errors.values()) < 1e-9, (family,name,errors)
            cases.append(dict(family=family,case=name,errors=errors,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(status='PASS',cases=cases,scope='Independent NumPy/SciPy preprocessing, time encoding, all 12 blocks and head versus untouched pinned source; exact current SDPA scaling, float64 CPU; no teaching implementation imported.',checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((root/'_sources_l068_v2.json').read_bytes()).hexdigest(),torch=torch.__version__)
    (root.parent/'reviews/lesson-quality-audit-047-070/068-independent.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(check(),indent=2))
