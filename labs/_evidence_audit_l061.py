"""Recompute L061 saved predictions with NumPy/SciPy, without lesson operators."""
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np
from scipy.special import logsumexp, erf, ndtri, ndtr
from scipy.stats import t

ROOT = Path(__file__).resolve().parent


def decode(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64'], validate=True))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    return np.frombuffer(raw, dtype=item['dtype']).reshape(item['shape']).astype(float)


def check(path, report_path=None):
    data = json.loads(path.read_text())
    assert data['status'] == 'COMPLETE'
    config = data['config']
    expected = {(seed, regime, n) for seed in config['seeds']
                for regime in ('matched', 'prior_shift') for n in config['contexts']}
    assert len(data['records']) == len(expected)
    assert {(r['seed'], r['regime'], r['n_context']) for r in data['records']} == expected
    borders = np.array(data['borders'])
    widths = np.diff(borders)
    assert np.all(widths > 0)
    centers = (borders[1:] + borders[:-1]) / 2
    scales = widths[[0, -1]] / ndtri(.75)
    centers[0] = borders[1] - scales[0] * math.sqrt(2 / math.pi)
    centers[-1] = borders[-2] + scales[-1] * math.sqrt(2 / math.pi)
    maximum = {}
    identities = {}
    head_records = []
    for row in data['records']:
        n = row['n_context']; lengthscale = .6 if row['regime'] == 'matched' else .1
        reconstructed = {key: [] for key in row['task_values']}
        head_losses = []
        key = (row['regime'], n)
        task_identity = [(a['x']['sha256'], a['y']['sha256']) for a in row['audit_batches']]
        if key in identities:
            assert task_identity == identities[key], 'Training seeds must share evaluation tasks'
        identities[key] = task_identity
        for audit in row['audit_batches']:
            a = {name: decode(value) for name, value in audit.items()}
            x, y, logits = a['x'], a['y'], a['logits'][:, 0]
            target = y[:, n]
            assert x.shape == (len(y), n + 1, config['features'])
            covariance = np.exp(-np.square(x[:, :, None] - x[:, None, :]).sum(-1)
                                / (2 * lengthscale ** 2))
            covariance += 1e-4 * np.eye(n + 1)
            if n:
                cross = covariance[:, n, :n]
                solved = np.linalg.solve(covariance[:, :n, :n],
                                         np.stack([y[:, :n], cross], -1))
                mean = np.sum(cross * solved[:, :, 0], -1)
                variance = 1.0001 - np.sum(cross * solved[:, :, 1], -1)
            else:
                mean = np.zeros(len(y)); variance = np.full(len(y), 1.0001)
            np.testing.assert_allclose(mean, a['gp_mean'][:, 0], atol=1e-8, rtol=0.)
            np.testing.assert_allclose(variance, a['gp_variance'][:, 0], atol=1e-10, rtol=0.)
            log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
            probabilities = np.exp(log_probs)
            index = np.searchsorted(borders, target, side='left') - 1
            index = np.clip(index, 0, len(widths) - 1)
            log_density = log_probs[np.arange(len(y)), index] - np.log(widths[index])
            for bucket, anchor, direction, scale in (
                    (0, borders[1], -1, scales[0]),
                    (len(widths) - 1, borders[-2], 1, scales[-1])):
                chosen = index == bucket
                distance = direction * (target[chosen] - anchor)
                log_density[chosen] = (log_probs[chosen, bucket] + .5 * math.log(2 / math.pi)
                                       - math.log(scale) - .5 * (distance / scale) ** 2)
            gp_cdf = ndtr((borders[None, 1:-1] - mean[:, None]) / np.sqrt(variance[:, None]))
            masses = np.diff(np.concatenate([np.zeros((len(y), 1)), gp_cdf,
                                            np.ones((len(y), 1))], axis=-1), axis=-1)
            head_log_mass = np.log(np.maximum(masses, 1e-300))
            head_log_mass -= logsumexp(head_log_mass, axis=-1, keepdims=True)
            head_losses.extend((-log_density + log_probs[np.arange(len(y)), index]
                                - head_log_mass[np.arange(len(y)), index]).tolist())
            fractions = np.clip((target[:, None] - borders[:-1]) / widths, 0, 1)
            fractions[:, 0] = 1 - erf(np.maximum(borders[1] - target, 0) / (scales[0] * math.sqrt(2)))
            fractions[:, -1] = erf(np.maximum(target - borders[-2], 0) / (scales[-1] * math.sqrt(2)))
            cdf = np.sum(probabilities * fractions, -1)
            gp_nll = .5 * (np.log(2 * math.pi * variance) + (target - mean) ** 2 / variance)
            outputs = dict(pfn_nll=-log_density, gp_nll=gp_nll,
                           prior_nll=.5 * (math.log(2 * math.pi * 1.0001) + target ** 2 / 1.0001),
                           mean_squared_error=(probabilities @ centers - mean) ** 2,
                           coverage95=((cdf >= .025) & (cdf <= .975)).astype(float),
                           excess_nll=-log_density - gp_nll)
            for name, value in outputs.items():
                reconstructed[name].extend(value.tolist())
        for name, values in reconstructed.items():
            assert len(values) == config['eval_tasks']
            error = float(np.max(np.abs(np.array(values) - row['task_values'][name])))
            maximum[name] = max(maximum.get(name, 0.), error)
            tolerance = 2e-5 if name not in ('coverage95', 'gp_nll', 'prior_nll') else (0 if name == 'coverage95' else 1e-7)
            assert error <= tolerance, (row['seed'], key, name, error)
            assert abs(np.mean(row['task_values'][name]) - row[name]) < 1e-12
        if row['seed'] == config['seeds'][0]:
            head_records.append(dict(regime=row['regime'], n_context=n,
                                     head_oracle_nll=float(np.mean(head_losses))))
    for summary in data['summary']:
        rows = [r for r in data['records'] if (r['regime'], r['n_context'])
                == (summary['regime'], summary['n_context'])]
        for metric in rows[0]['task_values']:
            tasks = np.array([r['task_values'][metric] for r in rows]).mean(0)
            means = np.array([r[metric] for r in rows])
            actual = summary[metric]
            assert abs(actual['mean'] - means.mean()) < 1e-12
            assert actual['seed_values'] == means.tolist()
            if len(rows) > 1:
                assert abs(actual['seed_sd'] - means.std(ddof=1)) < 1e-12
            halfwidth = t.ppf(.975, len(tasks) - 1) * tasks.std(ddof=1) / math.sqrt(len(tasks))
            assert abs(actual['conditional_task_t95'] - halfwidth) < 1e-12
    head_error = None
    analysis_path = ROOT / '_analysis_l061_v2_results.json'
    if path.name == '_verify_l061_v2_results.json' and analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        assert analysis['evidence_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        expected_head = {(r['regime'], r['n_context']): r for r in analysis['records']}
        assert set(expected_head) == {(r['regime'], r['n_context']) for r in head_records}
        head_error = max(abs(r['head_oracle_nll'] - expected_head[(r['regime'], r['n_context'])]['head_oracle_nll'])
                         for r in head_records)
        assert head_error < 1e-7
    report = dict(status='PASS', records=len(data['records']),
                  predictions=len(data['records']) * config['eval_tasks'],
                  maximum_metric_error=maximum,
                  max_fixed_head_oracle_error=head_error,
                  evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope='Raw tensor hashes, complete panel, paired task identities, NumPy GP solves, '
                        'SciPy full-support densities, means, coverage and all summary intervals; '
                        'float32 predictive-head arithmetic checked within 2e-5')
    output = Path(report_path) if report_path else ROOT.parent / 'reviews/lesson-quality-audit-047-070/061-evidence.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    check(ROOT / '_verify_l061_v2_results.json')
