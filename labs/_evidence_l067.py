"""Independent raw-data, retrieval, selection and original-model prediction audit.

Uses the released LoCalPFN PFN class, not the lesson model, loader or operators.
Adapted checkpoint bytes are local artifacts; their hashes are in public evidence.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

def digest(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode()); h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def source_name(name):
    if name.startswith('x_encoder.'): return 'encoder.' + name[10:]
    if name.startswith('head.'): return 'decoder.' + name[5:]
    return (name.replace('blocks.', 'transformer_encoder.')
            .replace('.qkv.weight', '.self_attn.in_proj_weight')
            .replace('.qkv.bias', '.self_attn.in_proj_bias')
            .replace('.out.', '.self_attn.out_proj.')
            .replace('.ff1.', '.linear1.').replace('.ff2.', '.linear2.'))

def raw_data(name):
    if name == 'wdbc':
        d = load_breast_cancer(); x, y = d.data, d.target
    else:
        d = pd.read_parquet(ROOT / f'data/cache/{name}.parquet')
        y = d.pop('class' if name == 'diabetes' else 'Class').to_numpy(); x = d.to_numpy()
    return x.astype('float32'), np.unique(y, return_inverse=True)[1]

def nearest(x, q, k, exclude=None):
    distances = np.sum((q[:, None] - x[None]) ** 2, axis=-1)
    if exclude is not None: distances[np.arange(len(q)), exclude] = np.inf
    return np.argsort(distances, axis=1, kind='stable')[:, :k]

def check(path, report, dataset_filter=None):
    data = json.loads(path.read_text()); cfg = data['config']; torch.set_num_threads(1)
    manifest = json.loads((ROOT / '_sources_l067_v2.json').read_text())
    source = ROOT / 'data/cache/l067-source/official/pfn.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['pfn.py']['sha256']
    spec = importlib.util.spec_from_file_location('l067_parent_original', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checkpoint = ROOT / 'data/cache/l067-source/official/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == data['checkpoint_sha256']
    state, _, c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    initial = {k.removeprefix('module.').replace('layers.', ''): v for k, v in state.items() if not k.endswith('criterion.weight')}
    model = module.PFN(dropout=c['dropout'], embedding_normalization=False, n_out=c['max_num_classes'],
                       nhead=c['nhead'], nhid=c['emsize']*c['nhid_factor'], ninp=c['emsize'],
                       nlayers=c['nlayers'], norm_first=False, num_features=c['num_features']).eval()
    records = []; count = 0; prediction_cache = {}
    def predict(x, y, q, contexts, classes):
        key = (digest(model.state_dict()), hashlib.sha256(x.tobytes()+y.tobytes()+q.tobytes()+contexts.tobytes()).hexdigest(), classes)
        if key in prediction_cache:return prediction_cache[key].copy()
        outputs = []
        with torch.no_grad():
            if np.all(contexts == contexts[0]):
                ids = contexts[0]
                xx = np.concatenate([x[ids], q])
                inp = torch.nn.functional.pad(torch.tensor(xx), (0, 100-x.shape[1]))[:, None]
                labels = torch.tensor(np.concatenate([y[ids], np.zeros(len(q))]), dtype=torch.float32)[:, None]
                result = model(inp, labels, len(ids), True, False, False, x.shape[1])[:, 0, :classes].softmax(-1).numpy()
                prediction_cache[key] = result.copy(); return result
            for start in range(0, len(q), cfg['inference_batch']):
                ids = contexts[start:start+cfg['inference_batch']]
                xx = np.concatenate([x[ids], q[start:start+len(ids), None]], axis=1)
                inp = torch.nn.functional.pad(torch.tensor(xx), (0, 100-x.shape[1])).transpose(0, 1).contiguous()
                labels = torch.tensor(np.concatenate([y[ids], np.zeros((len(ids), 1))], axis=1).T, dtype=torch.float32)
                out = model(inp, labels, ids.shape[1], True, False, False, x.shape[1])
                outputs.append(out[0, :, :classes].softmax(-1).numpy())
        result = np.concatenate(outputs); prediction_cache[key] = result.copy(); return result
    expected = {(d, s) for d in cfg['datasets'] for s in cfg['seeds']}
    assert {(r['dataset'], r['seed']) for r in data['records']} == expected
    assert len(data['records']) == len(expected)
    if dataset_filter is not None:assert dataset_filter in cfg['datasets']
    for r in data['records']:
        if dataset_filter is not None and r['dataset'] != dataset_filter:continue
        name, seed = r['dataset'], r['seed']; raw, y = raw_data(name)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == data['datasets'][name]['x_sha256']
        assert hashlib.sha256(y.tobytes()).hexdigest() == data['datasets'][name]['y_sha256']
        train, hold = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
        valid, test = train_test_split(hold, test_size=.5, random_state=seed, stratify=y[hold])
        for key, ids in [('train_ids', train), ('valid_ids', valid), ('test_ids', test)]: assert np.array_equal(ids, r[key])
        scaler = StandardScaler().fit(raw[train])
        assert np.array_equal(scaler.mean_, r['geometry']['mean']) and np.array_equal(scaler.scale_, r['geometry']['scale'])
        x, v, t = [np.clip(scaler.transform(raw[ids]), -10, 10).astype('float32') for ids in [train, valid, test]]
        yt = y[train]; classes = len(np.unique(yt)); assert classes == 2
        k = min(int(10*np.sqrt(len(train))), cfg['max_context'], len(train)-1); assert k == r['k']
        vc, tc = nearest(x, v, k), nearest(x, t, k)
        assert np.array_equal(train[vc], r['valid_context_ids']) and np.array_equal(train[tc], r['test_context_ids'])
        assert np.array_equal(y[test], r['targets']) and np.array_equal(y[valid], r['valid_targets'])
        random = np.random.default_rng(seed).permutation(len(train))[:k]
        assert np.array_equal(train[random], r['random_context_ids'])
        rng = np.random.default_rng(seed+67000); episodes = []
        for step in range(cfg['steps']):
            anchors = rng.choice(len(train), size=cfg['batch'], replace=False)
            n = nearest(x, x[anchors], k+r['queries_per_episode'], anchors)
            n = n[:, rng.permutation(n.shape[1])]
            episodes.append(dict(step=step+1, anchor_ids=train[anchors].tolist(), context_ids=train[n[:, :k]].tolist(), query_ids=train[n[:, k:]].tolist()))
        deltas = {}; validation_audit = {}; model.load_state_dict(initial)
        baseline = predict(x, yt, v, vc, classes)
        for arm, a in r['arms'].items():
            if arm in r['adaptation']:
                audit = r['adaptation'][arm]; assert audit['episodes'] == episodes
                assert len(audit['losses']) == cfg['steps'] and np.isfinite(audit['losses']).all()
                loaded = {}
                for role in ['final', 'selected']:
                    wp = Path(audit[role+'_weight_file'])
                    assert hashlib.sha256(wp.read_bytes()).hexdigest() == audit[role+'_weight_file_sha256']
                    loaded[role] = torch.load(wp, map_location='cpu', weights_only=True)
                    wanted = audit['final_weights_sha256'] if role == 'final' else a['selected_weights_sha256']
                    assert digest(loaded[role]) == wanted
                trace = audit['validation']; assert trace[0]['step'] == 0
                source_auc = [float(roc_auc_score(y[valid], baseline[:, 1]))]; validation_audit[arm] = []
                assert abs(trace[0]['validation_auc']-roc_auc_score(y[valid], baseline[:, 1])) < 1e-12
                for entry in trace[1:]:
                    assert entry['step'] == cfg['steps'], 'Checker currently supports measured 0/final schedule'
                    model.load_state_dict({source_name(n): value for n, value in loaded['final'].items()})
                    pv = predict(x, yt, v, vc, classes)
                    saved_valid = np.asarray(entry['validation_probabilities'], dtype='float32')
                    pd = float(np.max(abs(pv-saved_valid))); assert pd < 2e-4
                    assert abs(roc_auc_score(y[valid], saved_valid[:, 1])-entry['validation_auc']) < 1e-12
                    source_auc.append(float(roc_auc_score(y[valid], pv[:, 1])))
                    validation_audit[arm].append(dict(step=entry['step'], saved_auc=entry['validation_auc'], original_auc=source_auc[-1], probability_max_delta=pd, saved_p1_range=[float(saved_valid[:,1].min()),float(saved_valid[:,1].max())], original_p1_range=[float(pv[:,1].min()),float(pv[:,1].max())]))
                choice = int(np.argmax([entry['validation_auc'] for entry in trace]))
                assert int(np.argmax(source_auc)) == choice, (name, seed, arm, 'original-source validation choice differs', source_auc, trace)
                assert a['selected_step'] == audit['selected_step'] == trace[choice]['step']
                assert digest(loaded['selected']) == trace[choice]['weights_sha256']
                assert digest(loaded['final']) == trace[-1]['weights_sha256']
                delta = sum(float((value.double()-initial[source_name(n)].double()).square().sum()) for n, value in loaded['final'].items()) ** .5
                assert abs(delta-audit['final_parameter_l2_change']) < 1e-9 and delta > 0
                model.load_state_dict({source_name(n): value for n, value in loaded['selected'].items()})
            else: model.load_state_dict(initial)
            ids = tc if arm.startswith('local_') else np.tile(np.arange(len(train)) if arm=='global_all' else random, (len(test), 1))
            original = predict(x, yt, t, ids, classes); saved = np.asarray(a['probabilities'], dtype='float32')
            delta = float(np.max(abs(original-saved))); assert delta < 2e-4, (name, seed, arm, delta)
            assert abs(log_loss(y[test], saved)-a['log_loss']) < 1e-7
            assert accuracy_score(y[test], saved.argmax(1)) == a['accuracy']
            assert roc_auc_score(y[test], saved[:, 1]) == a['auc']
            deltas[arm] = delta; count += len(test)
        records.append(dict(dataset=name, seed=seed, predictions=len(test)*len(r['arms']), source_probability_max_deltas=deltas, validation=validation_audit))
        print(name, seed, 'PASS', json.dumps(validation_audit), flush=True)
    result = dict(status='PASS', evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), predictions=count, records=records, dataset_filter=dataset_filter,
                  scope='Independent raw data, split/scaler/neighbor/episode reconstruction, saved adapted weight hashes, validation selection, original PFN probabilities and deployed scores; discarded-state AUC may differ under near-constant float32 scores, but selected step is independently required to agree; adaptation trajectory itself not retrained')
    report.write_text(json.dumps(result, indent=2)+'\n'); return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('evidence', type=Path); p.add_argument('--report', type=Path, required=True); p.add_argument('--dataset')
    a = p.parse_args(); print(json.dumps(check(a.evidence, a.report, a.dataset)))
