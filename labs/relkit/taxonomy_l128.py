"""Task contracts and metrics: live learner implementations, no model wrapper."""
import math

def task_contract(kind):
    """Return the chosen released baseline contract; time is a separate axis."""
    choices = {
        'binary': ('one_logit', 'BCEWithLogits', 'roc_auc', True),
        'regression': ('one_scalar', 'L1', 'mae', False),
        'recommendation': ('candidate_scores', 'BPR', 'map', True),
    }
    if kind not in choices:
        raise ValueError('Use binary, regression or recommendation; time is orthogonal')
    head, loss, metric, maximize = choices[kind]
    return dict(head=head, loss=loss, metric=metric, maximize=maximize, split='temporal')

def binary_auc(labels, scores):
    """Probability a positive outranks a negative, with half credit for ties."""
    if len(labels) != len(scores) or not all(y in (0, 1) for y in labels):
        raise ValueError('Aligned binary labels required')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Finite scores required')
    positive = [float(s) for y, s in zip(labels, scores) if y == 1]
    negative = [float(s) for y, s in zip(labels, scores) if y == 0]
    if not positive or not negative:
        raise ValueError('AUROC requires both classes')
    return math.fsum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))

def ranking_map(truth, ranked, k):
    """Released RelBench macro MAP@k: exclude queries with no true destinations."""
    if not isinstance(k, int) or k < 1 or len(truth) != len(ranked):
        raise ValueError('Positive k and aligned queries required')
    values = []
    for relevant, row in zip(truth, ranked):
        relevant = set(relevant)
        if len(row) != k or len(set(row)) != k:
            raise ValueError('Exactly k unique ranked candidates required')
        if not relevant:
            continue
        hits = 0
        terms = []
        for rank, candidate in enumerate(row, 1):
            if candidate in relevant:
                hits += 1
                terms.append(hits / rank)
        values.append(math.fsum(terms) / min(k, len(relevant)))
    if not values:
        raise ValueError('No queries with positive destinations')
    return math.fsum(values) / len(values)
