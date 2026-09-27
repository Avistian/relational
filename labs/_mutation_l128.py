"""Ensure pedagogical checks reject plausible semantic mistakes."""
import json
from pathlib import Path
from _check_l128 import check_contract,check_auc,check_map
from relkit.taxonomy_l128 import task_contract,binary_auc,ranking_map
cases=[('minimize_AUROC',check_contract,lambda x:{**task_contract(x),'maximize':False}),
 ('wrong_recommendation_loss',check_contract,lambda x:{**task_contract(x),'loss':'BCEWithLogits'}),
 ('reverse_AUROC',check_auc,lambda y,p:1-binary_auc(y,p)),
 ('threshold_before_AUROC',check_auc,lambda y,p:binary_auc(y,[float(v>.5) for v in p])),
 ('use_hit_rate_for_MAP',check_map,lambda truth,ranked,k:sum(bool(set(t)&set(r)) for t,r in zip(truth,ranked))/len(truth)),
 ('rank_insensitive_MAP',check_map,lambda truth,ranked,k:ranking_map(truth,[sorted(r) for r in ranked],k))]
rejected=[]
for name,check,fn in cases:
 try:check(fn)
 except (AssertionError,ValueError):rejected.append(name)
 else:raise AssertionError('Surviving mutation: '+name)
r=dict(status='PASS',rejected=rejected)
Path(__file__).with_name('_mutation_l128_results.json').write_text(json.dumps(r,indent=2));print(r)
