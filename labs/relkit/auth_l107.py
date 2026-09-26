"""Authenticate all cached Wikipedia model inputs, independently of ZIP container bytes."""
import hashlib
from pathlib import Path
import numpy as np
WIKI_ARRAY_SHA={'u': 'a1cb9c71dd46635329ef7eec033ab3f2aaea186bda2439604eec0ce2ddf27baf', 'v': '9f13a20d8428ea30d197d9ad2bdc25153f293e76d432e09f67c93731af36439f', 't': '77b8cb3b9cbea5dad71cb90839a596c7e2d12713ee8df36b33765e7459e35ebc', 'x': '4edb58b6e9420fb687757b919d25de1fb62f9cc5cf89bd198b0b1945b153250f'}

def authenticate_wiki_cache(directory):
    path=Path(directory)/'processed.npz'
    if not path.exists():return
    with np.load(path,allow_pickle=False) as z:
        for key,expected in WIKI_ARRAY_SHA.items():
            observed=hashlib.sha256(z[key].tobytes()).hexdigest()
            if observed!=expected:raise ValueError('Untrusted Wikipedia cached field: '+key)
