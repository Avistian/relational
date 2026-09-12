"""Bind final executed teacher evidence to independently reconstructed author results."""
import hashlib,json,sys
from pathlib import Path
import nbformat,numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'labs'));from _archive_l070 import loss,task
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((r/'labs/_execution_l070_v2_results.json').read_text());assert e['status']=='PASS'
for key,p in [('notebook_sha256',e['notebook_path']),('executor_sha256',r/'labs/_execute_l070.py'),('exit_sha256',e['exit_path'])]:assert e[key]==sha(p)
nb=nbformat.read(e['notebook_path'],4);assert e['notebook_code_sha256']==hashlib.sha256(json.dumps([c.source for c in nb.cells if c.cell_type=='code'],sort_keys=True).encode()).hexdigest()
author=r/'labs/_verify_l070_v2_results.json';a=json.loads(author.read_text());t=json.loads(Path(e['exit_path']).read_text());proof=r/'reviews/lesson-quality-audit-047-070/070-evidence.json';pr=json.loads(proof.read_text());assert pr['status']=='PASS' and pr['evidence_sha256']==sha(author) and pr['checker_sha256']==sha(r/'labs/_evidence_l070.py')
assert t['status']=='COMPLETE' and len(a['records'])==len(t['records'])==30
ignore={'seconds','fit_selection_seconds','predict_seconds','lifecycle_100_batches_seconds','runtime_sha256'}
def stable(x):
 if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in ignore}
 if isinstance(x,list):return [stable(v) for v in x]
 return x
assert stable(a['records'])==stable(t['records']),'Fresh teacher science differs from independently refitted author'
assert a['datasets']==t['datasets'] and a['config']==t['config']
for row in t['records']:
 x,y,parts,digest,labels=task(row['dataset'].split('/')[0]);assert y[parts['test']].tolist()==row['targets']
 assert abs(loss(row['targets'],row['predictions'])-row['error'])<1e-12
 assert abs(loss(row['targets'],row['intervention']['predictions'])-row['intervention']['error'])<1e-12
 assert abs(row['lifecycle_100_batches_seconds']-(row['fit_selection_seconds']+100*row['predict_seconds']))<1e-9
 for c in row['candidates']:assert abs(loss(row['validation_targets'],c['validation_predictions'])-c['validation_error'])<1e-12
out=dict(status='PASS',execution=e,author_sha256=sha(author),independent_author_refit_report_sha256=sha(proof),records=30,candidates=60,validation_probability_rows=sum(len(c['validation_predictions']) for row in t['records'] for c in row['candidates']),test_and_intervention_probability_rows=sum(2*len(row['predictions']) for row in t['records']),runtime_identities=dict(author=sorted({c['model_identity']['runtime_sha256'] for row in a['records'] for c in row['candidates'] if 'runtime_sha256' in c['model_identity']}),teacher=sorted({c['model_identity']['runtime_sha256'] for row in t['records'] for c in row['candidates'] if 'runtime_sha256' in c['model_identity']})),scope='Actual final executed notebook/executor/EXIT/code hashes; every teacher candidate/history/model-weight/preprocessing/selection/test/intervention field exactly equals independently refitted author evidence, excluding measured timers and separately recorded runtime-code identities (module versus live notebook namespace). Raw target/row identities and all losses independently reconstructed; lifecycle arithmetic checked with teacher timers. No second independent teacher refit claimed.')
(r/'reviews/lesson-quality-audit-047-070/070-teacher-evidence.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='execution'})
