"""Reconstruct L063 evidence without importing its generator or posterior operators."""
import argparse, base64, hashlib, json, zlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent

def array(value):
    if isinstance(value, dict) and value.get('encoding') == 'zlib-base64':
        return np.frombuffer(zlib.decompress(base64.b64decode(value['data'])), dtype=value['dtype']).reshape(value['shape'])
    return np.asarray(value)

def activate(x, name):
    if name == 'identity': return x
    if name == 'tanh': return np.tanh(x)
    if name == 'leaky_relu': return np.maximum(x, 0) + .01 * np.minimum(x, 0)
    if name == 'elu': return np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
    raise AssertionError(name)

def forward(world, episode, edge=None, zero_noise=False):
    h = array(episode['causes']); outputs = []
    for layer, (raw, bias) in enumerate(zip(world['weights'], world['biases'])):
        w = array(raw).copy()
        if edge is not None and layer == len(world['weights']) - 1: w[edge] = 0
        incoming = h if layer == 0 else activate(h, world['activation'])
        # Sum each parent's contribution explicitly, independently of the live matrix product.
        h = np.column_stack([sum(incoming[:, parent] * weight for parent, weight in enumerate(row)) + b for row, b in zip(w, array(bias))])
        if layer and not zero_noise: h += array(episode['noises'][layer - 1])
        outputs.append(h)
    return outputs

def check(path, report_path=None):
    path = Path(path); result = json.loads(path.read_text()); cases = []
    for record in result['records']:
        world, episode = record['world'], record['episode']
        outputs = forward(world, episode)
        values = np.concatenate(outputs[1:], axis=1) if record['family'] == 'SCM' else np.column_stack([array(episode['causes']), outputs[-1][:, 0]])
        roles = episode['roles']; features = roles['feature_nodes']; target = roles['target_node']
        assert target not in features and len(set(features)) == len(features)
        x, z = values[:, features], values[:, target]
        delta = max(float(np.max(np.abs(x - array(episode['x'])))), float(np.max(np.abs(z - array(episode['continuous_target'])))))
        np.testing.assert_allclose(x, array(episode['x']), rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(z, array(episode['continuous_target']), rtol=1e-10, atol=1e-8)
        # Rank ties use saved exact generator outputs, avoiding an independent sum's roundoff.
        exact_z = array(episode['continuous_target']); bounds = exact_z[array(episode['bound_indices'])]
        y = array(episode['class_permutation'])[(exact_z[:, None] > bounds).sum(axis=1)]
        np.testing.assert_array_equal(y, array(episode['y']))
        np.testing.assert_array_equal(bounds, array(episode['bounds']))
        n = record['n_context']; truth = y[n:]; k = episode['classes']
        np.testing.assert_array_equal(truth, array(record['targets']))
        # Refit using only saved context rows; query rows never fit the scaler or coefficients.
        probability_errors = []
        for shuffled, field, metric in [(False, 'probabilities', 'nll'), (True, 'shuffled_probabilities', 'shuffled_nll')]:
            train_y = y[:n].copy()
            if shuffled: train_y = train_y[np.random.default_rng(record['seed'] + 100000).permutation(n)]
            p = np.zeros((len(truth), k)); observed = np.unique(train_y)
            if len(observed) == 1: p[:, observed[0]] = 1
            else:
                scaler = StandardScaler().fit(array(episode['x'])[:n])
                with threadpool_limits(limits=1):
                    model = LogisticRegression(C=1, max_iter=1000).fit(scaler.transform(array(episode['x'])[:n]), train_y)
                    p[:, model.classes_] = model.predict_proba(scaler.transform(array(episode['x'])[n:]))
            p = np.clip(p, 1e-12, 1); p /= p.sum(axis=1, keepdims=True)
            wanted = array(record[field]); np.testing.assert_allclose(p, wanted, atol=1e-12, rtol=1e-12)
            loss = -np.log(wanted[np.arange(len(truth)), truth]).mean()
            assert abs(loss - record[metric]) < 1e-12
            probability_errors.append(float(np.abs(p-wanted).max()))
        intervention = record['intervention']; edge = (intervention['child'], intervention['parent'])
        weight = array(world['weights'][-1])[edge]
        assert weight == intervention['removed_weight']
        assert intervention['nonzero_edge'] == (weight != 0)
        changed = forward(world, episode, edge=edge)
        actual = changed[-1][:, edge[0]] - outputs[-1][:, edge[0]]
        wanted = -activate(outputs[-2][:, edge[1]], world['activation']) * weight
        # Subtracting large affine values can expose roundoff when the edge effect is small.
        # Bound the two sums and subtraction by their absolute contributions, not their difference.
        terms = activate(outputs[-2], world['activation']) * array(world['weights'][-1])[edge[0]]
        scale = np.abs(terms).sum(axis=1) + abs(array(world['biases'][-1])[edge[0]]) + np.abs(array(episode['noises'][-1])[:,edge[0]])
        rounding_bound = 128*np.finfo(float).eps*np.maximum(1,scale)
        oracle_residual = np.abs(actual-wanted)
        saved_residual = np.abs(actual-array(intervention['delta']))
        assert np.all(oracle_residual <= rounding_bound)
        assert np.all(saved_residual <= rounding_bound)
        assert all(np.array_equal(a,b) for a,b in zip(outputs[:-1],changed[:-1])) and intervention['unchanged_upstream']
        noiseless = forward(world, episode, zero_noise=True)
        noise_mae = float(np.abs(noiseless[-1] - outputs[-1]).mean())
        assert np.isclose(noise_mae, intervention['structural_noise_last_layer_mae'], atol=1e-8, rtol=1e-10)
        missing = sorted(set(truth.tolist()) - set(y[:n].tolist()))
        cases.append(dict(family=record['family'], seed=record['seed'], rows=len(y), forward_max_abs=delta,
                          probability_max_abs=max(probability_errors), query_classes_absent_from_context=missing,
                          nonzero_intervention=bool(weight != 0),
                          intervention_max_rounding_bound_fraction=float(np.max(np.maximum(oracle_residual,saved_residual)/rounding_bound))))
    for summary in result['summary']:
        rows = [r for r in result['records'] if r['family'] == summary['family']]
        gaps = np.array([r['shuffled_nll'] - r['nll'] for r in rows])
        assert len(rows) == summary['tasks']
        for key, value in [('nll', np.mean([r['nll'] for r in rows])), ('shuffled_nll', np.mean([r['shuffled_nll'] for r in rows])), ('paired_gap_mean', gaps.mean())]:
            assert abs(summary[key] - value) < 1e-12
        if len(rows)>1: assert abs(summary['paired_gap_sd']-gaps.std(ddof=1)) < 1e-12
        assert summary['positive_gaps'] == int((gaps>0).sum())
    oracle = result['finite_oracle']; errors=[]; losses=[]; ignored_losses=[]
    for r in oracle['records']:
        # Gaussian X likelihood ratio is 2x; Bernoulli Y ratio is (2y-1) log 9.
        context_odds = 2*sum(r['cx']) + (2*sum(r['cy'])-len(r['cy']))*np.log(9)
        w = expit(context_odds + 2*r['qx']); p=.1+.8*w; ignored=.1+.8*expit(context_odds)
        errors.append(max(abs(w-r['weights'][1]), abs(p-r['p']), abs(ignored-r['ignored_p'])))
        losses.append(-np.log(p if r['qy'] else 1-p)); ignored_losses.append(-np.log(ignored if r['qy'] else 1-ignored))
    assert max(errors)<1e-12
    assert abs(np.mean(losses)-oracle['correct_nll'])<1e-12
    assert abs(np.mean(ignored_losses)-oracle['ignored_query_x_nll'])<1e-12
    report=dict(status='PASS', evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), records=len(cases), cases=cases,
                oracle_cases=len(errors), oracle_max_abs=max(errors),
                scope='Independent parent-contribution sums, observation/label reconstruction, context-only diagnostic refits, interventions, metrics and closed-form posterior odds. No full historical prior or PFN training reproduction.')
    if report_path: Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('path',nargs='?',default=ROOT/'_verify_l063_v2_results.json'); p.add_argument('--report',default=ROOT.parent/'reviews/lesson-quality-audit-047-070/063-independent.json'); a=p.parse_args()
    print(json.dumps(check(a.path,a.report),indent=2))
