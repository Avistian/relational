"""B19 evaluation evidence: course diagnostics, not a TFM implementation."""
import itertools
import math
import numpy as np


def split_audit(rows, train_ids, test_ids):
    """Audit row identity separately from group separation; overlap is evidence."""
    by_id = {r['id']: r for r in rows}
    train, test = set(train_ids), set(test_ids)
    if len(by_id) != len(rows) or len(train) != len(train_ids) or len(test) != len(test_ids):
        raise ValueError('Duplicate row identity')
    if not train or not test or train & test or not (train | test) <= by_id.keys():
        raise ValueError('Empty, overlapping or unknown row IDs')
    overlap = sorted({by_id[i]['group'] for i in train} & {by_id[i]['group'] for i in test})
    return dict(train_n=len(train), test_n=len(test), overlap_groups=overlap, group_disjoint=not overlap)


def paired_scores(rows, models, policy='measured'):
    """Common measured support, or explicit RF substitution; never silent fill."""
    if policy not in ('measured', 'rf') or not models or len(set(models)) != len(models):
        raise ValueError('Invalid policy or model list')
    cells = {}
    for r in rows:
        key = (r['dataset'], r['seed'], r['model'])
        if key in cells:
            raise ValueError('Duplicate score identity')
        loss, origin = r['loss'], r['origin']
        if (loss is None and origin != 'missing') or (loss is not None and (origin != 'measured' or not math.isfinite(loss))):
            raise ValueError('Invalid score or origin')
        cells[key] = r
    keys = sorted({(r['dataset'], r['seed']) for r in rows})
    selected, used = [], []
    for dataset, seed in keys:
        current = []
        for model in models:
            row = cells.get((dataset, seed, model))
            if row is None:
                raise ValueError('Missing identity: declare missing runs explicitly')
            if row['loss'] is None:
                if policy == 'measured':
                    break
                rf = cells.get((dataset, seed, 'RF'))
                if rf is None or rf['loss'] is None:
                    raise ValueError('No measured RF substitute')
                row = dict(row, loss=rf['loss'], origin='imputed:RF')
            current.append(dict(row))
        if len(current) == len(models):
            selected.append([dataset, seed]); used.extend(current)
    if not selected:
        raise ValueError('No common support')
    means = {m: float(np.mean([r['loss'] for r in used if r['model'] == m])) for m in models}
    return dict(keys=selected, cells=used, means=means, order=sorted(models, key=lambda m: (means[m], m)))


def dataset_summary(rows):
    """Equal weight per dataset after averaging seeds; seed rows are dependent."""
    if not rows:
        raise ValueError('No datasets')
    seen, grouped = set(), {}
    for r in rows:
        key = (r['dataset'], r['seed'])
        if key in seen or not math.isfinite(r['delta']):
            raise ValueError('Duplicate or nonfinite paired difference')
        seen.add(key); grouped.setdefault(r['dataset'], []).append(r['delta'])
    means = {d: float(np.mean(v)) for d, v in sorted(grouped.items())}
    n = len(means)
    se = float(np.std(list(means.values()), ddof=1) / np.sqrt(n)) if n > 1 else None
    naive = float(np.std([r['delta'] for r in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None
    return dict(n_datasets=n, n_seed_rows=len(rows), dataset_means=means, mean=float(np.mean(list(means.values()))), se_dataset=se, se_naive_rows=naive)


def make_fixture(seed):
    rng = np.random.default_rng(seed)
    labels = rng.permutation([0]*12 + [1]*12)
    rows = [dict(id=8*g+j, group=g, y=int(labels[g]), signal=int(labels[g] if rng.random()<.7 else 1-labels[g])) for g in range(24) for j in range(8)]
    random_test = sorted(int(8*g+j) for g in range(24) for j in rng.choice(8, 2, replace=False))
    held_groups = set(int(g) for g in rng.choice(24, 8, replace=False))
    grouped_test = [r['id'] for r in rows if r['group'] in held_groups]
    return dict(seed=seed, rows=rows, tests=dict(random=random_test, grouped=grouped_test))


def run_experiment():
    fixtures, arms = [], []
    for seed in (0,1,2):
        f = make_fixture(seed); fixtures.append(f)
        for regime, test in f['tests'].items():
            train = [r['id'] for r in f['rows'] if r['id'] not in set(test)]
            audit = split_audit(f['rows'], train, test)
            train_rows = [f['rows'][i] for i in train]
            prevalence = float(np.mean([r['y'] for r in train_rows]))
            memory = {g: float(np.mean([r['y'] for r in train_rows if r['group']==g])) for g in {r['group'] for r in train_rows}}
            for model in ('group_memory','signal'):
                predictions = [dict(id=i, group=f['rows'][i]['group'], y=f['rows'][i]['y'], p=memory.get(f['rows'][i]['group'],prevalence) if model=='group_memory' else .2+.6*f['rows'][i]['signal']) for i in test]
                # Training-only predictor values; query labels enter scoring only.
                loss = float(np.mean([(p['p']-p['y'])**2 for p in predictions]))
                arms.append(dict(seed=seed, regime=regime, model=model, train_ids=train, audit=audit, predictions=predictions, brier=loss))
    score_rows = [dict(dataset=f'd{d}',seed=0,model=m,loss=v,origin='missing' if v is None else 'measured') for d,vals in enumerate(zip([.1,.1,.1,None],[.2]*4,[.25,.25,.25,.9])) for m,v in zip(['A','B','RF'],vals)]
    deltas = [dict(dataset=d,seed=s,delta=v) for d,values in [('a',[.10,.11,.09]),('b',[-.04,-.05,-.03]),('c',[.02,.03,.01])] for s,v in enumerate(values)]
    summary = dataset_summary(deltas)
    dataset_boot = sorted(float(np.mean(v)) for v in itertools.product(summary['dataset_means'].values(), repeat=3))
    row_three = sorted(float(np.mean(v)) for v in itertools.product([r['delta'] for r in deltas], repeat=3))
    duplicated = dataset_summary(deltas + [dict(r,seed=r['seed']+3) for r in deltas])
    return dict(protocol='B19-EVIDENCE-BOUNDARIES-v1',fixtures=fixtures,arms=arms,score_rows=score_rows,policies={p:paired_scores(score_rows,['A','B','RF'],p) for p in ('measured','rf')},deltas=deltas,uncertainty=summary,duplicated_seeds=duplicated,dataset_bootstrap=dataset_boot,three_row_resampling=row_three,paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')


if __name__ == '__main__':
    import json
    print(json.dumps(run_experiment(),indent=2))
