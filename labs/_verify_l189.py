"""Independent Fraction oracle, malformed-packet and learner-mutation checks."""
import itertools,json,random,shutil,tempfile
from fractions import Fraction
from pathlib import Path
from _audit_l189 import audit
from _test_l189 import checks
from relkit.gaps_l189 import admission,priority,sensitivity
P=Path(__file__).resolve().parent;E=P/'evidence/l189';Q=E/'packet'
m=json.loads((E/'input-manifest.json').read_text());r=json.loads((E/'report.json').read_text());cases=json.loads((Q/'cases.json').read_text())
assert r==audit(Q,m,admission,priority,sensitivity)
for row in r['weight_grid']:
 w=row['weights'];truth={c['id']:Fraction(c['impact']*sum(a*b for a,b in zip(c['feasibility'],w)),sum(w)) for c in cases}
 assert row['leaders']==sorted(k for k,v in truth.items() if v==max(truth.values()))
 for k,v in truth.items():assert row['scores'][k]==float(v)
rng=random.Random(189)
for _ in range(200):
 i=rng.randint(1,5);f=[rng.randint(1,5) for _ in range(3)];w=[rng.randint(1,3) for _ in range(3)]
 assert priority(i,f,w)==float(Fraction(i*sum(a*b for a,b in zip(f,w)),sum(w)))
checks(admission,priority,sensitivity)
for fns in [(lambda p:{'status':'WITHIN_CAP','total_usd':0},priority,sensitivity),(admission,lambda i,f,w:sum(f),sensitivity),(admission,priority,lambda c:sensitivity(c)[:1])]:
 try:checks(*fns)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Wrong learner function passed')
# Even with a newly computed manifest, unsupported semantic claims fail.
for field,value in [('novelty','ESTABLISHED'),('baseline',''),('related_work',['missing'])]:
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'packet';shutil.copytree(Q,q);cc=json.loads((q/'cases.json').read_text());cc[0][field]=value;(q/'cases.json').write_text(json.dumps(cc))
  import hashlib
  mm={'files':{str(p.relative_to(q)):hashlib.sha256(p.read_bytes()).hexdigest() for p in q.rglob('*') if p.is_file()}}
  try:audit(q,mm,admission,priority,sensitivity)
  except ValueError:pass
  else:raise AssertionError('Unsupported semantics passed')
result=dict(status='PASS',fraction_scenarios=27,random_score_cases=200,wrong_learner_functions_rejected=3,semantic_corruptions_rejected=3,report_parity='EXACT',scope='Report audit and rubric only, not prediction/model replay')
(P/'_verify_l189_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
