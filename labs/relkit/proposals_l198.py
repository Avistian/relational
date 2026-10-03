"""Visible proposal mechanics. Scores are matched AUROC examples, not new model results."""
import itertools
import math


def paired_contrast(scores, mode):
    """A,B = baseline/intervention; C,D = baseline/intervention under second policy/prior."""
    required={'gain':{'a','b'},'conditional':{'a','b','c','d'},'interaction':{'a','b','c','d'}}
    if mode not in required or set(scores)!=required[mode]:
        raise ValueError('Supply exactly the arms required by the contrast')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise ValueError('AUROC values must be finite numbers in [0,1]')
    if mode=='gain':return scores['b']-scores['a']
    if mode=='conditional':return scores['d']-scores['c']
    return (scores['d']-scores['c'])-(scores['b']-scores['a'])


def interval_decision(low, high, margin, mode):
    """Classify an externally justified interval; this function does not estimate uncertainty."""
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in [low,high,margin]):
        raise ValueError('Finite numeric interval and margin required')
    if low>high or margin<=0 or mode not in ['benefit','sensitivity']:
        raise ValueError('Ordered interval, positive margin and known mode required')
    if mode=='benefit':
        if low>margin:return 'USEFUL_BENEFIT'
        if high<margin:return 'BELOW_USEFUL_MARGIN'
    else:
        if low>margin or high<-margin:return 'MATERIAL_SENSITIVITY'
        if low>-margin and high<margin:return 'BELOW_USEFUL_MARGIN'
    return 'INCONCLUSIVE'


def expand_matrix(axes):
    """Enumerate every declared combination in insertion order; never hide seeds or controls."""
    if not isinstance(axes,dict) or not axes:raise ValueError('Nonempty named axes required')
    for name,values in axes.items():
        if not isinstance(name,str) or not name.strip() or not isinstance(values,list) or not values:
            raise ValueError('Each axis needs a name and nonempty list')
        if any(type(v) not in (str,int) for v in values) or len(set(values))!=len(values):
            raise ValueError('Distinct string/integer levels required; booleans are not seed IDs')
    return [dict(zip(axes,values)) for values in itertools.product(*axes.values())]
