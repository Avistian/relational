"""L058 frozen-table analysis and held-out-method benchmark selection.

This is an independently implemented, specified interpretation of TALENT §8.2,
not the authors' original selector or a regeneration of Table 7.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ARMS = ['xgboost', 'catboost', 'mlp', 'realmlp', 'tabr', 'ftt']
PAPER_POOL = ['dummy', 'LR', 'knn', 'svm', 'xgboost', 'catboost',
              'RandomForest', 'mlp', 'resnet', 'ftt', 'dcn2', 'tabr', 'tabpfn']
FILES = {'binary': 'cls_bin.md', 'multiclass': 'cls_multi.md', 'regression': 'regression.md'}


def parse_talent(path):
    """Return means, SDs as aligned frames; reject malformed nonmissing cells."""
    lines = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip().startswith('|')]
    if len(lines) < 3:
        raise ValueError('Expected header, separator and data rows')
    split = lambda line: [v.strip() for v in line.strip('|').split('|')]
    columns = split(lines[0])
    if columns[0].lower() != 'dataset' or len(set(columns)) != len(columns):
        raise ValueError('Require unique model columns and Dataset header')
    if len(split(lines[1])) != len(columns) or not all(
            re.fullmatch(r':?-+:?', v) for v in split(lines[1])):
        raise ValueError('Invalid Markdown separator')
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    means, sds, ids = [], [], []
    for line in lines[2:]:
        cells = split(line)
        if len(cells) != len(columns) or not cells[0] or cells[0] in ids:
            raise ValueError('Malformed row or duplicate dataset identity')
        ids.append(cells[0]); mu = []; sd = []
        for raw in cells[1:]:
            value = raw.replace('*', '').strip()
            if value in {'', '-', '—', 'NA', 'N/A', 'nan+nan'}:
                mu.append(np.nan); sd.append(np.nan); continue
            match = re.fullmatch(f'({number})\\s*(?:\\+|±)\\s*({number})', value)
            if not match:
                raise ValueError(f'Invalid mean+SD cell on {cells[0]}: {raw}')
            a, b = map(float, match.groups())
            if not np.isfinite([a, b]).all() or b < 0:
                raise ValueError('Nonfinite mean or invalid SD')
            mu.append(a); sd.append(b)
        means.append(mu); sds.append(sd)
    return (pd.DataFrame(means, index=ids, columns=columns[1:]),
            pd.DataFrame(sds, index=ids, columns=columns[1:]))


def rank_matrix(scores, higher_is_better):
    """Rank only the supplied model pool, one finite complete dataset per row."""
    x = np.asarray(scores, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty complete finite dataset-by-model matrix')
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise ValueError('Declare the metric direction explicitly')
    return rankdata(-x if higher_is_better else x, axis=1, method='average')


def dataset_bootstrap(matrix, draws=2000, seed=58):
    """Paired dataset percentile intervals; output [2, models], not t intervals."""
    x = np.asarray(matrix, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty dataset rows')
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws < 1:
        raise ValueError('draws must be a positive integer')
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(0, len(x), (draws, len(x)))].mean(axis=1)
    return np.quantile(sampled, [.025, .975], axis=0)


def select_tiny(seen_ranks, size, trials=1000, seed=58):
    """Choose the first minimum-MAE random subset; never receives unseen scores."""
    x = np.asarray(seen_ranks, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected a finite seen-method rank matrix')
    for value in (size, trials):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError('size and trials must be positive integers')
    if size > len(x):
        raise ValueError('Cannot select more unique datasets than available')
    rng = np.random.default_rng(seed); target = x.mean(axis=0)
    best = None; best_error = np.inf; history = []; first = None
    for _ in range(trials):
        ids = np.sort(rng.choice(len(x), size, replace=False))
        if first is None:
            first = ids.copy()
        error = float(np.abs(x[ids].mean(axis=0) - target).mean())
        if error < best_error:
            best, best_error = ids.copy(), error
        history.append(best_error)
    return dict(indices=best.tolist(), first_indices=first.tolist(),
                seen_mae=best_error, best_so_far=history)


def method_holdout(scores, higher_is_better, seen, size, trials=1000, seed=58):
    """Rerank BEFORE selection within seen pool; score unseen pool only afterward."""
    x = np.asarray(scores, float)
    rank_matrix(x, higher_is_better)  # validate complete panel
    ix = np.asarray(seen)
    if ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix) or len(set(ix)) != len(ix):
        raise ValueError('seen must be unique integer model indices')
    if ix.min() < 0 or ix.max() >= x.shape[1] or len(ix) == x.shape[1]:
        raise ValueError('Require nonempty disjoint seen and unseen pools')
    unseen = [j for j in range(x.shape[1]) if j not in set(ix)]
    sr = rank_matrix(x[:, ix], higher_is_better)
    selection = select_tiny(sr, size, trials, seed)
    ur = rank_matrix(x[:, unseen], higher_is_better)
    result = dict(seen=ix.tolist(), unseen=unseen, selection=selection)
    for label, indices in [('random', selection['first_indices']), ('selected', selection['indices'])]:
        result[label] = {name: float(np.abs(r[indices].mean(0) - r.mean(0)).mean())
                         for name, r in [('seen_mae', sr), ('unseen_mae', ur)]}
    return result


def load_tables(root):
    root = Path(root); source = json.loads((root / '_sources_l058.json').read_text())
    tables = {}; dispersions = {}
    for task, filename in FILES.items():
        path = root / 'sources/l058' / filename
        expected = source['files']['results/' + filename]['sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned TALENT table hash mismatch')
        means, sd = parse_talent(path)
        tables[task] = means.rename(columns={'LogReg': 'LR', 'LinearRegression': 'LR'})
        dispersions[task] = sd.rename(columns={'LogReg': 'LR', 'LinearRegression': 'LR'})
    return tables, dispersions, source


def audit(root, trials=1000, seeds=(58, 59, 60)):
    tables, sd, source = load_tables(root)
    full = []; coverage = {}; groups = {}; row_ids = []
    for task, frame in tables.items():
        common = frame[ARMS].dropna()
        coverage[task] = dict(released=len(frame), complete=len(common),
                             excluded=frame.index[frame[ARMS].isna().any(axis=1)].tolist())
        ranks = rank_matrix(common.to_numpy(), task != 'regression')
        groups[task] = dict(zip(ARMS, ranks.mean(0).tolist()))
        full.extend(ranks); row_ids.extend([f'{task}:{i}' for i in common.index])
    full = np.asarray(full); ci = dataset_bootstrap(full)
    gaps = full[:, 1:2] - full[:, [3, 4]]  # CatBoost minus RealMLP / TabR
    gap_ci = dataset_bootstrap(gaps)
    tiny = []; tiny_coverage = {}
    for task, frame in tables.items():
        pool = [m for m in PAPER_POOL if m in frame.columns]
        common = frame[pool].dropna(); x = common.to_numpy()
        tiny_coverage[task] = dict(released=len(frame), complete=len(common),
            size=round(.15 * len(common)), pool=pool,
            excluded=frame.index[frame[pool].isna().any(axis=1)].tolist())
        # Three held out for classification, two for regression; five declared
        # deterministic rotations, not claimed to recover the paper's splits.
        width = 2 if task == 'regression' else 3
        for fold in range(5):
            unseen = [(fold * width + j) % len(pool) for j in range(width)]
            seen = [j for j in range(len(pool)) if j not in unseen]
            for seed in seeds:
                result = method_holdout(x, task != 'regression', seen,
                                        round(.15 * len(x)), trials, seed)
                result.update(task=task, fold=fold, seed=seed, pool=pool,
                              selected_ids=common.index[result['selection']['indices']].tolist())
                # Preserve the curve compactly at fixed computation budgets.
                history = result['selection'].pop('best_so_far')
                result['selection']['trace'] = {str(i): history[i - 1] for i in [1, 10, 100, 1000, 10000] if i <= trials}
                tiny.append(result)
    return dict(scope='Fresh frozen-table reanalysis; no training', source_revision=source['revision'],
                arms=ARMS, coverage=coverage, task_ids=row_ids, mean_ranks=dict(zip(ARMS, full.mean(0).tolist())),
                per_task_ranks=groups, percentile95=ci.tolist(), bootstrap=dict(draws=2000, seed=58, unit='dataset'),
                paired_gaps={name: dict(mean=float(gaps[:, j].mean()), percentile95=gap_ci[:, j].tolist())
                             for j, name in enumerate(['catboost-minus-realmlp', 'catboost-minus-tabr'])},
                tiny_trials=trials, tiny_coverage=tiny_coverage, tiny=tiny, verdict='INCOMPARABLE',
                omissions='Frozen historical release; independent within-group ranks; declared new method splits; no original Table 7 regeneration')
