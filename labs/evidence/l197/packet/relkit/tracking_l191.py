"""L191 published-table arithmetic; never executes a model."""
import math

def signed_gap(target, comparator, metric):
    """Positive means Kumo is better; AUROC inputs use published percent units."""
    if metric not in ('AUROC', 'MAE') or not all(math.isfinite(x) for x in (target, comparator)):
        raise ValueError('Require finite values and AUROC or MAE')
    return target - comparator if metric == 'AUROC' else comparator - target

def eligible(metadata, pool):
    """Eligibility is an audited policy, not a claim of full reproducibility."""
    if pool not in ('foundation', 'supervised', 'all'):
        raise ValueError('Unknown pool')
    return (metadata.get('access') == 'open_code'
            and metadata.get('protocol') == 'reported_same_table'
            and metadata.get('family') in ('foundation', 'supervised')
            and (pool == 'all' or metadata['family'] == pool))

def compare_pool(target, candidates, metric, baseline=None):
    """Single method and descriptive taskwise oracle on complete common coverage.

    Selection uses published test scores: retrospective summary, not a deployment
    selection rule. For MAE aggregate ratios before averaging, never raw MAEs.
    """
    if not target or metric not in ('AUROC', 'MAE'):
        raise ValueError('Empty target or unknown metric')
    n = len(target)
    if any(not math.isfinite(v) for v in target):
        raise ValueError('Nonfinite target')
    if metric == 'MAE':
        if baseline is None or len(baseline) != n or any(not math.isfinite(v) or v <= 0 for v in baseline):
            raise ValueError('Require positive LightGBM baseline per task')
    scale = baseline if metric == 'MAE' else [1.0] * n
    complete, excluded = {}, []
    for name, values in candidates.items():
        if len(values) != n:
            raise ValueError('Task coverage length mismatch')
        if any(v is not None and not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite candidate')
        if any(v is None for v in values):
            excluded.append(name)
        else:
            complete[name] = values
    if not complete:
        return dict(status='NO_ELIGIBLE_COMPARATOR', excluded=sorted(excluded))
    scores = {k: sum(v / b for v, b in zip(values, scale)) / n for k, values in complete.items()}
    choose = max if metric == 'AUROC' else min
    best = choose(scores.values())
    winners = sorted(k for k, v in scores.items() if math.isclose(v, best, rel_tol=0, abs_tol=1e-12))
    oracle = [choose(v[i] for v in complete.values()) for i in range(n)]
    oracle_methods = [sorted(k for k, v in complete.items() if v[i] == oracle[i]) for i in range(n)]
    target_score = sum(v / b for v, b in zip(target, scale)) / n
    oracle_score = sum(v / b for v, b in zip(oracle, scale)) / n
    return dict(status='COMPLETE_PUBLISHED_COMPARISON', single_methods=winners,
                single_score=best, target_score=target_score,
                single_gap=signed_gap(target_score, best, metric), oracle_values=oracle,
                oracle_methods=oracle_methods, oracle_score=oracle_score,
                oracle_gap=signed_gap(target_score, oracle_score, metric),
                per_task_gaps=[signed_gap(a, b, metric) for a, b in zip(target, oracle)],
                excluded=sorted(excluded))

def average_ranks(values, higher):
    """Midranks of displayed values; hidden precision cannot be recovered."""
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Require nonempty finite scores')
    return [1.0 + sum((w > v if higher else w < v) for w in values)
            + (sum(w == v for w in values) - 1) / 2 for v in values]
