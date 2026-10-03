"""Small visible evidence contracts for the Year 5 writing unit."""
import math


def coverage_report(expected, rows):
    """Require exactly the declared task set; missing scores stay missing."""
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Expected task IDs must be unique and nonempty')
    seen = {}
    for row in rows:
        task, status, score = row['task'], row['status'], row['score']
        if task in seen or task not in expected:
            raise ValueError('Duplicate or undeclared task')
        if status == 'NOT_RUN':
            if score is not None:
                raise ValueError('An unrun task cannot have a score')
        elif status == 'MEASURED':
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('Measured scores must be finite numbers')
        else:
            raise ValueError('Unknown result status')
        seen[task] = status
    if set(seen) != set(expected):
        raise ValueError('Incomplete declared inventory')
    missing = [task for task in expected if seen[task] == 'NOT_RUN']
    return dict(declared=len(expected), measured=len(expected)-len(missing), missing=missing)


def admit_claim(lane, claim, authenticated, complete):
    """A teaching policy for THIS packet, not an automated scientific referee."""
    policy = {'published_table': {'published_comparison'},
              'saved_predictions': {'scoped_pipeline_comparison'},
              'source_diagnostic': {'implementation_observation'}}
    known = set.union(*policy.values()) | {'fresh_model_reproduction', 'architecture_cause', 'economic_undervaluation'}
    if lane not in policy or claim not in known:
        raise ValueError('Unknown evidence lane or claim')
    if type(authenticated) is not bool or type(complete) is not bool:
        raise ValueError('Evidence flags must be booleans')
    return 'ADMISSIBLE_SCOPED' if authenticated and complete and claim in policy[lane] else 'NOT_ESTABLISHED'


def essay_readiness(sections):
    """Check five required fields for presence only; do not grade prose quality."""
    required = ['claim', 'evidence', 'warrant', 'limitation', 'revision']
    if any(k not in sections or not isinstance(sections[k], str) for k in required):
        raise ValueError('Five textual argument fields required')
    missing = [k for k in required if not sections[k].strip()]
    return dict(state='DRAFT' if missing else 'READY_FOR_REVIEW', missing=missing,
                mastery='PENDING_WRITTEN_DEFENSE')
