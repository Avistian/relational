"""Behavioral acceptance: paired pool, live functions and corrected model dependency."""
import inspect
import numpy as np
from relkit.temporal_v2 import paired_random_split
from relkit.temporal_experiment_v2 import load_release, TabM
parts={'train':np.array([90,2,7,8]),'val':np.array([11,19]),'test':np.array([24,32])}
r=paired_random_split(parts,55)
assert {k:len(v) for k,v in r.items()}=={k:len(v) for k,v in parts.items()}
np.testing.assert_array_equal(np.sort(np.concatenate(list(r.values()))),np.sort(np.concatenate(list(parts.values()))))
assert all(np.array_equal(r[k],paired_random_split(parts,55)[k]) for k in r)
assert not all(np.array_equal(r[k],parts[k]) for k in r)
assert list(inspect.signature(paired_random_split).parameters)==['temporal_ids','seed']
try: paired_random_split({'train':[1,2],'val':[2],'test':[3]},0)
except ValueError: pass
else: raise AssertionError('Overlapping row IDs must be rejected')
for name in ('ecom-offers','homesite-insurance','sberbank-housing'):
    a=load_release(name,'random',root='labs/data/cache/l055')[-1]
    b=load_release(name,'temporal',root='labs/data/cache/l055')[-1]
    assert a['pool_hash']==b['pool_hash'] and a['sizes']==b['sizes']
assert TabM.__module__=='relkit.tabm_v2'
m=TabM(6,k=3,depth=2)
assert all(b.S is None and b.bias.ndim==1 for b in m.blocks)
assert m.blocks[1].R is None
print('PASS: same pool/counts, deterministic label-free assignment, overlap rejection, corrected mini')
