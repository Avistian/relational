"""Independent brute-force path enumeration and semantic mutants."""
import itertools,json,random
from pathlib import Path
import numpy as np
from relkit.scale_l134 import frontier_bound,audit_queries,profile_summary
from _check_l134 import check_bound,check_audit,check_summary
rng=random.Random(134)
for trial in range(100):
    schema=[(a,f'r{i}',b) for i,(a,b) in enumerate(itertools.product('abc',repeat=2)) if rng.random()<.35]
    fanouts=[rng.randrange(4),rng.randrange(4)];batch=rng.randrange(1,5)
    # Explicit expansion of occurrence tokens; no recurrence shared with implementation.
    tokens=['a']*batch;total=len(tokens)
    for f in fanouts:
        tokens=[src for dst in tokens for src,_,receiver in schema if receiver==dst for _ in range(f)];total+=len(tokens)
    assert frontier_bound(schema,'a',batch,fanouts)['node_occurrences']==total
mutants=[(check_bound,lambda s,k,b,f:dict(frontiers=[],node_occurrences=b*(1+sum(f)),edge_occurrences=b*sum(f))),
 (check_audit,lambda t,o,c,e:dict(nodes=sum(len(v) for v in o.values()),edges=0)),
 (check_summary,lambda r:dict(queries=sum(x['queries'] for x in r),queries_per_second=sum(x['queries']/(x['sample_s']+x['step_s']) for x in r)/len(r),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in r),audited_queries_per_second=0))]
for check,mutant in mutants:
    try:check(mutant)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Mutation survived')
r=dict(status='PASS',independent_expansion_cases=100,rejected_mutants=3)
Path(__file__).with_name('_verify_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
