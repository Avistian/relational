"""Independent expected tables, not calls back into the implementation."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
j=json.loads((p/'evidence/b15/diagnostic.json').read_text())
assert len(j['rules'])==4 and len(j['columns'])==8 and len(j['interventions'])==16
assert {(r['theta'],r['b']) for r in j['rules']}=={(0,0),(0,1),(1,0),(1,1)}
for r in j['rules']:
 truth=int(r['theta']!=r['b']);assert r['truth']==truth
 assert r['local']==.5 and r['expanded']==truth and r['leaky']==truth
assert {(r['s'],r['a'],r['b']) for r in j['columns']}=={(s,a,b) for s in (0,1) for a in (0,1) for b in (0,1)}
for r in j['columns']:
 assert r['local']==(r['a']+r['b'])/2
 assert r['expanded']==r['truth']==([r['a'],r['b']][r['s']])
for r in j['interventions']:
 assert r['safe']==int(r['theta']!=r['b']) and r['leaked']==r['query_label']
for group,key in [('rule_scores','rules'),('column_scores','columns')]:
 for arm,score in j[group].items():
  rows=j[key];correct=sum((r[arm]>=.5)==bool(r['truth']) for r in rows)
  assert score['accuracy']==correct/len(rows)
  assert score['brier']==sum((r[arm]-r['truth'])**2 for r in rows)/len(rows)
assert j['information_bits']=={'local':0.0,'expanded':1.0}
result=dict(status='PASS',rule_worlds=4,column_worlds=8,interventions=16,scope=j['scope'])
(p/'_verify_b15_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
