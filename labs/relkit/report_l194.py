"""Visible analysis contracts. Reference arithmetic never becomes fresh inference."""
import math
from collections import Counter


def oriented_gap(metric, model, comparator):
    """Positive means model better; missing results stay missing."""
    if metric not in {'AUROC', 'MAE'}:
        raise ValueError('Unknown metric')
    for value in (model, comparator):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Scores must be finite numbers or null')
            if value < 0 or (metric == 'AUROC' and value > 1):
                raise ValueError('Score outside metric domain')
    if model is None or comparator is None:
        return None
    return model - comparator if metric == 'AUROC' else comparator - model


def evidence_license(kind):
    """The strongest permitted statement for each evidence type in THIS packet."""
    licenses = {
        'PUBLISHED_TABLE': 'DESCRIPTIVE_REFERENCE_ONLY',
        'SAVED_SOURCE_DIAGNOSTIC': 'SOURCE_INVARIANT_FAILURE_ONLY',
        'UNRUN_BENCHMARK': 'NO_PERFORMANCE_CONCLUSION',
        'PROPOSED_INTERVENTION': 'HYPOTHESIS_NOT_TESTED',
    }
    if kind not in licenses:
        raise ValueError('Unknown or unsupported evidence type')
    return licenses[kind]


def summarize_comparisons(rows, expected_ids):
    """Require the declared task set; count directions without pooling units."""
    if len(set(expected_ids)) != len(expected_ids) or not expected_ids:
        raise ValueError('Expected IDs must be nonempty and unique')
    if Counter(r['task'] for r in rows) != Counter(expected_ids):
        raise ValueError('Missing, duplicate or unexpected task')
    counts = {'higher': 0, 'lower': 0, 'equal_at_printed_precision': 0, 'missing': 0}
    for row in rows:
        gap = row['gap']
        if gap is None:
            counts['missing'] += 1
        elif isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap):
            raise ValueError('Invalid gap')
        elif gap > 0:
            counts['higher'] += 1
        elif gap < 0:
            counts['lower'] += 1
        else:
            counts['equal_at_printed_precision'] += 1
    return dict(tasks=len(rows), **counts, scope='Descriptive signs; no pooled metric or significance claim')
