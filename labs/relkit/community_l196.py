"""Visible learner contracts for a reproducible community question."""
import math
from urllib.parse import urlsplit

def summarize_cases(cases):
    """Require all four interventions; never infer a model score from encoder codes."""
    order = ['a', 'z', '0', 'e']
    if len(cases) != 4 or {c['unseen'] for c in cases} != set(order):
        raise ValueError('Require each of a, z, 0, e exactly once')
    indexed = {c['unseen']: c for c in cases}
    changed, batching = [], []
    numeric_ok = True
    for unseen in order:
        case = indexed[unseen]
        for name, length in [('before', 3), ('after', 3), ('numeric_before', 3),
                             ('numeric_after', 3), ('query_alone', 1), ('query_with_other', 2)]:
            values = case[name]
            if len(values) != length or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
                raise ValueError('Malformed numeric observation: ' + name)
        if case['before'] != case['after']:
            changed.append(unseen)
        if case['query_alone'][0] != case['query_with_other'][1]:
            batching.append(unseen)
        numeric_ok = numeric_ok and case['numeric_before'] == case['numeric_after']
    return dict(cases=4, changed_cases=changed, batch_sensitive_cases=batching,
                numeric_controls_unchanged=numeric_ok, model_effect='NOT_ESTABLISHED')

def route_question(owner):
    """Course routing map, verified 2026-10-02; exact thread access requires an account."""
    routes = {'rdblearn': 'https://github.com/HKUSHXLab/rdblearn/issues',
              'relbench': 'https://huggingface.co/relbench',
              'pyg': 'https://github.com/pyg-team/pytorch_geometric/discussions'}
    if owner not in routes:
        raise ValueError('Identify the implementation owner first')
    return routes[owner]

def feedback_state(thread_url, response, verified):
    """A pedagogical state policy; a boolean is not proof of external verification."""
    if not isinstance(thread_url, str) or not isinstance(response, str) or type(verified) is not bool:
        raise ValueError('Require textual thread/response and boolean verification')
    if thread_url.strip():
        parsed = urlsplit(thread_url)
        if parsed.scheme != 'https' or not parsed.hostname:
            raise ValueError('Require a real HTTPS thread URL')
    else:
        return 'DRAFT_ONLY'
    if not response.strip():
        return 'AWAITING_RESPONSE'
    return 'FEEDBACK_CHECKED' if verified else 'RESPONSE_UNVERIFIED'
