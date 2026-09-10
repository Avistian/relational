"""L050 teaching additions; no change to the historical measured training recipe."""
import hashlib
import json
import types
import numpy as np
from _live_identity_l051 import code_fingerprint


def mean_predictions(probabilities):
    """Average selected binary probability vectors over models, preserving row order."""
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or not all(p.shape) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Need a nonempty finite [models, rows] probability matrix in [0,1].')
    return p.mean(axis=0)


def implementation_identity(namespace):
    """Fingerprint live functions, constructors, defaults and the shared data harness.

    Callable defaults are represented by name and their own executable content,
    never process-specific repr addresses. Locations and interpreter bookkeeping
    are excluded. Package versions/data/settings are recorded by the run operator.
    This covers the declared visible experiment, not arbitrary external monkeypatches.
    """
    from pathlib import Path
    names=['prepare_numeric','reglu','select_trial','paired_summary','train_neural','predict','run_comparison']
    objects={name:namespace[name] for name in names}
    for name in ['CheckpointFT','CheckpointBlock','CheckpointAttention','CheckpointMLP']:
        for method in ['__init__','forward']:
            objects[name+'.'+method]=getattr(namespace[name],method)
    if 'mean_predictions' in namespace: objects['mean_predictions']=namespace['mean_predictions']
    def value(x):
        if isinstance(x,types.FunctionType): return {'function':x.__name__,'body':function(x)}
        if isinstance(x,type): return {'class':x.__name__}
        if isinstance(x,(tuple,list)): return [value(v) for v in x]
        if isinstance(x,dict): return {k:value(v) for k,v in x.items()}
        if x is None or isinstance(x,(str,int,float,bool)): return x
        raise TypeError('Unsupported live default: '+type(x).__name__)
    def function(f):
        bare=types.FunctionType(f.__code__,f.__globals__,f.__name__,None,f.__closure__)
        return {'code':code_fingerprint(bare),'defaults':value(f.__defaults__),
                'kwdefaults':value(f.__kwdefaults__),
                'closure':value(tuple(c.cell_contents for c in (f.__closure__ or ())))}
    from relkit import data
    payload={'objects':{name:function(f) for name,f in objects.items()},
             'data_harness':hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
