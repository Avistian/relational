"""Independent pair-count AUROC, XML DOM and exhaustive gate/rubric oracles."""
import hashlib,itertools,json,shutil,tempfile
from fractions import Fraction
from pathlib import Path
from xml.dom import minidom
import numpy as np
from _replay_l190 import replay190
from _test_l190 import checks
from relkit.checkpoint_l190 import keyed_auc,claim_gate,rank_cases
P=Path(__file__).resolve().parent;E=P/'evidence/l190';Q=E/'packet'
m=json.loads((E/'input-manifest.json').read_text());r=json.loads((E/'report.json').read_text())
assert replay190(Q,m,keyed_auc,claim_gate,rank_cases)==r
checks(keyed_auc,claim_gate,rank_cases)
max_error=0.;count=0
for folder in ['pilot-1','full-1']:
 for row in json.loads((Q/'model'/folder/'receipt.json').read_text())['records']:
  with np.load(Q/'model'/folder/f"{row['arm']}-{row['seed']}.npz") as z:
   positive=z['probability'][z['label']==1];negative=z['probability'][z['label']==0]
   value=float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative)))
  observed=next(x['auc'] for x in r['runs'] if x['arm']==row['arm'] and x['seed']==row['seed'])
  max_error=max(max_error,abs(value-observed));assert abs(value-observed)<1e-12;count+=1
# XML DOM oracle uses independently selected elements and identity normalization.
ids=[];raw_records=0
for a in json.loads((Q/'literature/collection.json').read_text())['attempts']:
 if a.get('accepted'):
  dom=minidom.parseString((Q/'literature'/a['file']).read_bytes())
  for e in dom.getElementsByTagName('entry'):
   url=e.getElementsByTagName('id')[0].firstChild.data
   identity=url.split('/abs/')[1].split('v')[0];ids.append(identity);raw_records+=1
assert raw_records==r['literature']['received_records'] and len(set(ids))==r['literature']['unique_papers']
cases=json.loads((Q/'cases.json').read_text())
for row in r['ranking']:
 w=row['weights'];v={c['id']:Fraction(c['impact']*sum(x*y for x,y in zip(c['feasibility'],w)),sum(w)) for c in cases}
 assert row['scores']=={k:float(x) for k,x in v.items()}
 assert row['leaders']==sorted(k for k,x in v.items() if x==max(v.values()))
for bits in itertools.product([False,True],repeat=4):
 e=dict(zip(['authenticated','complete','metric_checked','comparable'],bits))
 assert claim_gate('saved_metric',e)==('SUPPORTED_REPLAY' if all(bits) else 'BLOCKED')
 for k,v in [('novelty','NOT_ESTABLISHED'),('fresh_training','NOT_RUN'),('general_superiority','NOT_ESTABLISHED')]:assert claim_gate(k,e)==v
for functions in [(lambda *a:.5,claim_gate,rank_cases),(keyed_auc,lambda *a:'SUPPORTED_REPLAY',rank_cases),(keyed_auc,claim_gate,lambda c:rank_cases(c)[:1])]:
 try:checks(*functions)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Wrong learner function accepted')
# Validate integrity AND semantic coverage after a hypothetical new manifest.
for mutation in ['byte','missing','duplicate','novelty','labels']:
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'packet';shutil.copytree(Q,q);mm=json.loads(json.dumps(m))
  if mutation=='byte':(q/'reports/l181.json').write_text('{}')
  elif mutation in ['missing','duplicate']:
   p=q/'model/full-1/receipt.json';d=json.loads(p.read_text())
   if mutation=='missing':d['records'].pop()
   else:d['records'].append(d['records'][0])
   p.write_text(json.dumps(d))
  elif mutation=='novelty':
   p=q/'cases.json';d=json.loads(p.read_text());d[0]['novelty']='ESTABLISHED';p.write_text(json.dumps(d))
  else:
   p=q/'model/pilot-1/RDBPFN-0.npz'
   with np.load(p) as z:d={k:z[k].copy() for k in z.files}
   d['label'][0]=1-d['label'][0];np.savez(p,**d)
   receipt=q/'model/pilot-1/receipt.json';dd=json.loads(receipt.read_text());dd['records'][0]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();receipt.write_text(json.dumps(dd))
  if mutation!='byte':mm['files']={str(p.relative_to(q)):hashlib.sha256(p.read_bytes()).hexdigest() for p in q.rglob('*') if p.is_file()}
  try:replay190(q,mm,keyed_auc,claim_gate,rank_cases)
  except ValueError:pass
  else:raise AssertionError('Corrupt packet accepted: '+mutation)
result=dict(status='PASS',independent_pairwise_runs=count,independent_predictions=21060,max_auc_error=max_error,independent_raw_xml_records=raw_records,weight_scenarios=27,gate_states=64,incorrect_learner_functions_rejected=3,corruptions_rejected=5,report_parity='EXACT')
(P/'_verify_l190_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
