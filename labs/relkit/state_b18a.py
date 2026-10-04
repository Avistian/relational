"""Visible, model-independent state and explanation operations. NumPy only."""
import hashlib
import json
import numpy as np


def eligible_support(rows, cutoff):
    """A label must actually be available at the prediction cutoff."""
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate support ID')
    return [r for r in rows if max(r['event'], r['available'], r['label_available']) <= cutoff]


def state_key(*, ids, x, y, weights, preprocessing, recipe):
    """Order-sensitive identity; include fitted preprocessing and full inference recipe.

    NaNs use an explicit mask and zero payload, avoiding NaN bit-pattern ambiguity.
    This detects changes; it does not itself invalidate a third-party cache.
    """
    x = np.asarray(x, dtype='<f8')
    y = np.asarray(y, dtype='<f8')
    if len(ids) != len(x) or len(y) != len(x) or len(set(ids)) != len(ids):
        raise ValueError('Misaligned or duplicate support identities')
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError('Invalid support values')
    h = hashlib.sha256()
    h.update(json.dumps(dict(ids=list(ids), shape=x.shape, weights=weights,
                            preprocessing=preprocessing, recipe=recipe),
                        sort_keys=True, allow_nan=False, separators=(',', ':')).encode())
    h.update(np.isnan(x).tobytes())
    h.update(np.nan_to_num(x, nan=0.).tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


def replacement_contrast(predict, x, background, feature):
    """Descriptive contrast, not SHAP: f(x) - mean_b f(x with x_j=b_j)."""
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    if x.ndim != 2 or background.ndim != 2 or not len(background) or x.shape[1] != background.shape[1]:
        raise ValueError('Nonempty compatible matrices required')
    if not 0 <= feature < x.shape[1]:
        raise ValueError('Unknown feature')
    modified = np.repeat(x, len(background), axis=0)
    modified[:, feature] = np.tile(background[:, feature], len(x))
    return np.asarray(predict(x)) - np.asarray(predict(modified)).reshape(len(x), len(background)).mean(axis=1)
