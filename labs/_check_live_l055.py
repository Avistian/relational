"""Changing a real live split/helper changes saved identity; execution does not."""
from _live_identity_l055 import live_identity
from relkit import temporal_experiment_v2 as e
from relkit.tabm_v2 import BatchEnsembleLinear,PackedHead
space=vars(e).copy();space.update(BatchEnsembleLinear=BatchEnsembleLinear,PackedHead=PackedHead)
a=live_identity(space)
space['paired_random_split']({'train':[0,1],'val':[2],'test':[3]},0)
assert live_identity(space)==a
# New live root routes through a helper; changing only that helper is observable.
exec('def local_helper(ids, seed):\n return {k:v for k,v in ids.items()}\ndef paired_random_split(temporal_ids, seed):\n return local_helper(temporal_ids, seed)',space)
b=live_identity(space);assert b!=a
exec('def local_helper(ids, seed):\n return {k:list(reversed(v)) for k,v in ids.items()}',space)
assert live_identity(space)!=b
assert '0x' not in str(live_identity(space))
exec('def local_helper(x):\n return x+1\ndef paired_random_split(temporal_ids,seed):\n return {k:tuple(local_helper(x) for x in v) for k,v in temporal_ids.items()}',space)
c=live_identity(space)
exec('def local_helper(x):\n return x+2',space)
assert live_identity(space)!=c, 'Nested generator helper must affect identity'
print('PASS: stable after execution; edited live split and reachable helper invalidate identity')
