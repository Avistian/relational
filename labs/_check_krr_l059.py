"""Independent centered-kernel reconstruction of L059's augmented-system KRR."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from relkit.validation_audit_l059 import krr_fit, predict_krr, loo_residuals, exact_null_minimum

ROOT = Path(__file__).resolve().parent


def reference(x, y, query, regularization, eta):
    # Centering the implicit feature map handles an unpenalized intercept.
    k = np.exp(-np.sum((x[:, None] - x[None, :]) ** 2 * eta, axis=-1))
    h = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    alpha = np.linalg.solve(h @ k @ h + regularization * np.eye(len(x)), y - y.mean())
    bias = y.mean() - k.mean(axis=0) @ alpha
    qk = np.exp(-np.sum((query[:, None] - x[None, :]) ** 2 * eta, axis=-1))
    return qk @ alpha + bias


def check(path=ROOT / '_verify_l059_v2_results.json'):
    # Integer binomial coefficients provide an independent survival-sum check.
    survival = np.array([sum(math.comb(80, j) for j in range(k + 1, 81)) / 2**80
                         for k in range(80)])
    null_checks = []
    for budget in [1, 4, 16, 64, 256]:
        expected = float(np.sum(survival**budget) / 80)
        actual = exact_null_minimum(80, budget)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        null_checks.append(dict(candidates=budget, expected_minimum=expected))
    rng = np.random.default_rng(59123)
    x = rng.normal(size=(12, 2)); x[1] = x[0]  # Duplicate inputs remain valid with lambda > 0.
    y = rng.choice([-1., 1.], len(x)); query = rng.normal(size=(7, 2))
    cases = []
    for lam in [.01, .1, 1.]:
        for eta in [.25, np.array([2., 16.]), np.array([16., .25])]:
            model = krr_fit(x, y, lam, eta)
            actual = predict_krr(model, query)
            expected = reference(x, y, query, lam, eta)
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
            deleted = np.array([y[i] - reference(np.delete(x, i, 0), np.delete(y, i),
                                x[i:i+1], lam, eta)[0] for i in range(len(x))])
            residuals = loo_residuals(model['alpha'], model['inverse_diagonal'])
            np.testing.assert_allclose(residuals, deleted, atol=2e-10, rtol=0)
            shifted = krr_fit(x, y + 3., lam, eta)
            np.testing.assert_allclose(predict_krr(shifted, query), actual + 3., atol=2e-10, rtol=0)
            cases.append(dict(regularization=lam, eta=np.asarray(eta).tolist(),
                              prediction_max_error=float(np.max(np.abs(actual-expected))),
                              loo_max_error=float(np.max(np.abs(residuals-deleted)))))
    r = json.loads(path.read_text())
    # Locate the experiment even when the runner wraps it with null reanalysis.
    experiment = r if 'records' in r else next(v for v in r.values() if isinstance(v, dict) and 'records' in v)
    errors = []
    for record in experiment['records']:
        xx, yy = np.array(record['x']), np.array(record['y'])
        internal = np.empty(len(xx)); external = np.empty(len(xx))
        covered = []
        for fold in record['trace']:
            fit, held = np.array(fold['fit_ids']), np.array(fold['held_ids'])
            assert not set(fit) & set(held)
            assert sorted(np.r_[fit, held]) == list(range(len(xx)))
            covered.extend(held.tolist())
            configs = experiment['config']['candidates']
            internal[held] = reference(xx[fit], yy[fit], xx[held], **configs[fold['selected']])
            external[held] = reference(xx[fit], yy[fit], xx[held], **configs[record['external_selected']])
        assert sorted(covered) == list(range(len(xx)))
        for name, prediction in [('internal', internal), ('external', external)]:
            np.testing.assert_allclose(prediction, record[name+'_predictions'], atol=2e-10, rtol=0)
            np.testing.assert_allclose(np.mean((yy-prediction)**2), record[name+'_mse'], atol=2e-10, rtol=0)
            errors.append(float(np.max(np.abs(prediction-record[name+'_predictions']))))
    out = dict(status='PASS', null_checks=null_checks, mechanism_cases=cases,
               artifact=path.name, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               recorded_repetitions=len(experiment['records']),
               recorded_oof_max_error=max(errors), source_sha256=hashlib.sha256(
                   (ROOT / 'relkit/validation_audit_l059.py').read_bytes()).hexdigest(),
               scope='Independent centered-kernel predictions and explicit deleted-row refits, target-shift/intercept check, every saved internal/external OOF prediction and MSE; no upstream software or full-paper parity')
    suffix = '-closer' if 'closer' in path.stem else ''
    (ROOT.parent / f'reviews/lesson-quality-audit-047-070/059-krr{suffix}.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=ROOT / '_verify_l059_v2_results.json')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        check(args.artifact)
