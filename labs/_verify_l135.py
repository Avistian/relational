"""Independent ranking enumeration and semantic mutants; trainer parity audit."""
import ast,itertools,json,random,statistics
from pathlib import Path
from _check_l135 import check_select,check_reserve,check_paired
from relkit.tuning_l135 import select_configuration,reserve_budget,paired_differences
P=Path(__file__).parent;rng=random.Random(135)
for _ in range(100):
 rows=[dict(config=c,seed=s,selection_mae=rng.randrange(10)) for c in 'abc' for s in [2,7]]
 rng.shuffle(rows)
 reference=min('abc',key=lambda c:(sum(r['selection_mae'] for r in rows if r['config']==c)/2,c))
 assert select_configuration(rows,list('abc'),[2,7])['winner']==reference
for check,mutant in [(check_select,lambda rows,c,s:dict(winner=min(rows,key=lambda r:r['selection_mae'])['config'])),(check_reserve,lambda p,w,t,r,o,c:sum(p)+w*t*r),(check_paired,lambda a,b:dict(differences=[y['mae']-x['mae'] for x,y in zip(a,b)],mean_difference=0))]:
 try:check(mutant)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Broken mechanism survived')
# Source delta is explicit and reviewable rather than claiming historical identity.
source=(P/'relkit/tuning_train_l135.py').read_text()
assert "splits = [\"train\",\"val\",\"test\"] if evaluate_test else [\"train\",\"val\"]" in source
assert "lr=CONFIG['lr']" in source and "num_neighbors=CONFIG['fanout']" in source
assert "if score<best:" in source
r=dict(status='PASS',independent_ranking_cases=100,rejected_semantic_mutants=3,source_contract='Configurable learning rate/fanout, audit-only additions; explicit test gate. Numerical source parity checked in each cloud fit.')
(P/'_verify_l135_results.json').write_text(json.dumps(r,indent=2));print(r)
