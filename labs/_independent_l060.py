"""Independent artifact audit: raw labels, scalar losses and pairwise rank counts.

Does not import the lesson runner, selector, metric or report implementations.
This verifies saved predictions and reporting, not how a model was trained.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGETS = {'diabetes': 'class', 'blood_transfusion': 'Class', 'kc1': 'defects',
           'phoneme': 'Class', 'credit_g': 'class', 'churn': 'class',
           'bank_marketing': 'Class', 'adult': 'class'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    # Count strictly smaller and equal opponents, independently of scipy.rankdata.
    return [1 + sum(other < value for other in values)
            + (sum(other == value for other in values) - 1) / 2 for value in values]


def check(path):
    data = json.loads(path.read_text())
    assert data['status'] == 'COMPLETE'
    for name, expected in data['source_hashes'].items():
        assert sha(ROOT / 'relkit' / name) == expected, ('source changed', name)
    raw_targets = {}
    for key, audit in data['datasets'].items():
        name, regime = key.split('/')
        ids = audit['ids']
        assert set(ids) == {'train', 'val', 'test'}
        flat = sum([ids[s] for s in ('train', 'val', 'test')], [])
        assert len(flat) == len(set(flat)) and all(ids.values())
        if name in TARGETS:
            cache = ROOT / 'data/cache' / (name + '.parquet')
            assert sha(cache) == audit['data_sha256']
            frame = pd.read_parquet(cache)
            target = frame[TARGETS[name]]
            if target.dtype == object or str(target.dtype) == 'category':
                labels = target.astype(str).tolist()
                positive = max(set(labels))
                full = np.array([int(v == positive) for v in labels])
            else:
                full = target.to_numpy(dtype=int)
        else:
            folder = ROOT / 'data/cache/l055' / name / name
            for file, expected in audit['data_hashes'].items():
                assert sha(folder / file) == expected
            full = np.load(folder / 'y.npy')
            for part in ids:
                released = np.load(folder / 'splits' / audit['split'] / (part + '.npy'))
                assert set(ids[part]).issubset(set(released.tolist()))
                array = np.array(ids[part], dtype=released.dtype)
                assert hashlib.sha256(array.tobytes()).hexdigest() == audit['index_hashes'][part]
        raw_targets[key] = full[ids['test']].tolist()
    by_key = {}
    max_error = 0.
    for row in data['records']:
        key = (row['dataset'], row['arm'], row['seed'])
        assert key not in by_key
        by_key[key] = row
        assert row['test_ids'] == data['datasets'][row['dataset']]['ids']['test']
        assert row['targets'] == raw_targets[row['dataset']]
        assert len(row['predictions']) == len(row['targets'])
        expected_choice = min(range(len(row['validation_errors'])),
                              key=row['validation_errors'].__getitem__)
        assert row['selected'] == expected_choice
        assert row['epoch'] == row['candidate_epochs'][expected_choice]
        pairs = list(zip(row['targets'], row['predictions']))
        if row['metric'] == 'RMSE':
            error = math.sqrt(math.fsum((y - p) ** 2 for y, p in pairs) / len(pairs))
        else:
            assert row['class_order'] == [0, 1]
            eps = np.finfo(float).eps
            loss = []
            for y, p in pairs:
                assert y in (0, 1) and 0 <= p <= 1
                p = min(max(p, eps), 1 - eps)
                loss.append(-math.log(p) if y else -math.log1p(-p))
            error = math.fsum(loss) / len(loss)
        max_error = max(max_error, abs(error - row['error']))
        assert math.isclose(error, row['error'], rel_tol=1e-11, abs_tol=1e-12)
        assert math.isclose(row['seconds'], row['fit_selection_seconds'] + row['predict_seconds'])
    arms, seeds = data['design']['arms'], data['config']['seeds']
    panels, rank_matrices = {}, {}
    expected = set()
    for regime, datasets in data['design']['panels'].items():
        matrix = []
        for dataset in datasets:
            means = []
            for arm in arms:
                keys = [(dataset, arm, seed) for seed in seeds]
                expected.update(keys)
                means.append(math.fsum(by_key[k]['error'] for k in keys) / len(seeds))
            matrix.append(ranks(means))
        matrix = np.array(matrix)
        rank_matrices[regime] = matrix
        means = dict(zip(arms, matrix.mean(0).tolist()))
        for arm in arms:
            assert math.isclose(means[arm], data['summary'][regime]['mean_ranks'][arm], abs_tol=1e-12)
        for dataset, row in zip(datasets, matrix):
            np.testing.assert_array_equal(row, data['summary'][regime]['dataset_ranks'][dataset])
        panels[regime] = {'datasets': len(datasets), 'mean_ranks': means}
        if len(matrix) == 3 and len(arms) == 5:
            # Simultaneous column relabeling leaves the statistic unchanged.
            # Fix row zero and enumerate all 5! x 5! remaining permutations.
            permutations = list(itertools.permutations(range(5)))
            observed = float(np.sum((matrix.sum(0) - 9.) ** 2))
            exceed = 0
            for p in permutations:
                for q in permutations:
                    sums = matrix[0] + matrix[1, list(p)] + matrix[2, list(q)]
                    exceed += float(np.sum((sums - 9.) ** 2)) >= observed - 1e-12
            panels[regime]['exact_exchangeable_rank_p'] = exceed / 14400
            panels[regime]['enumerated_permutations'] = 14400
    assert set(by_key) == expected
    analysis_checks = {}
    if path.name == '_verify_l060_v2_results.json':
        analysis_path = ROOT / '_analysis_l060_v2_results.json'
        analysis = json.loads(analysis_path.read_text())
        assert analysis['evidence_sha256'] == sha(path)
        assert analysis['summary'] == data['summary']
        # For three seeds, invert the closed-form t(2) CDF independently.
        tcrit = math.sqrt(2 * .95 ** 2 / (1 - .95 ** 2))
        for pair in analysis['paired_effects']:
            dataset, arm = pair['dataset'], pair['arm']
            delta = [by_key[(dataset, arm, seed)]['error']
                     - by_key[(dataset, 'XGBoost', seed)]['error'] for seed in sorted(seeds)]
            assert len(delta) == 3 and pair['differences'] == delta
            mean = math.fsum(delta) / 3
            sd = math.sqrt(math.fsum((d - mean) ** 2 for d in delta) / 2)
            half = tcrit * sd / math.sqrt(3)
            assert math.isclose(pair['mean'], mean, abs_tol=1e-12)
            assert math.isclose(pair['sd'], sd, abs_tol=1e-12)
            np.testing.assert_allclose(pair['t95'], [mean - half, mean + half], atol=1e-10)
        actual_pairs = {(p['dataset'], p['arm']) for p in analysis['paired_effects']}
        expected_pairs = {(d, a) for d in data['datasets'] for a in arms if a != 'XGBoost'}
        assert actual_pairs == expected_pairs and len(actual_pairs) == len(analysis['paired_effects'])
        temporal = analysis['permutation']['temporal']
        assert temporal['p'] == panels['temporal']['exact_exchangeable_rank_p']
        random = analysis['permutation']['random']
        matrix = rank_matrices['random']
        center = len(matrix) * (len(arms) + 1) / 2
        observed = sum((sum(row[j] for row in matrix) - center) ** 2 for j in range(len(arms)))
        rng = np.random.default_rng(random['seed'])
        exceed = 0
        for _ in range(random['draws']):
            permuted = [rng.permutation(row) for row in matrix]
            dispersion = sum((sum(row[j] for row in permuted) - center) ** 2 for j in range(len(arms)))
            exceed += dispersion >= observed - 1e-9
        assert exceed == random['exceed']
        assert random['p'] == (exceed + 1) / (random['draws'] + 1)
        analysis_checks = {'artifact_sha256': sha(analysis_path), 'paired_intervals': len(actual_pairs),
                           't2_quantile': tcrit, 'random_permutations': random['draws'],
                           'random_exceed': int(exceed), 'random_p': random['p'],
                           'temporal_p': temporal['p']}
    report = {'status': 'PASS', 'artifact': path.name, 'artifact_sha256': sha(path),
              'records': len(by_key), 'raw_label_dataset_regimes': len(raw_targets),
              'max_metric_error': max_error, 'panels': panels, 'analysis_checks': analysis_checks,
              'scope': 'Independent raw-label and row/hash alignment, scalar prediction losses, validation choices, explicit crossed roster and seed-first rank counts; temporal exact within-row method-label calibration assumes exchangeability; no training reproduction claim'}
    suffix = '' if path.name == '_verify_l060_v2_results.json' else '-smoke'
    out = ROOT.parent / f'reviews/lesson-quality-audit-047-070/060-independent{suffix}.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, default=ROOT / '_verify_l060_v2_results.json')
    check(parser.parse_args().artifact)
