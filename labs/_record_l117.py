"""Freeze verified local evidence without claiming learner mastery or live Colab."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
s=json.loads((P/'evidence/l117/summary.json').read_text());checks={}
for name in ['red','check','source_check','audit','mutation','execution','delivery']:
 p=P/f'_{name}_l117_results.json';v=json.loads(p.read_text());assert v['status']=='PASS';checks[p.name]='PASS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(json.loads(p.read_text())['source_sha256']==sha(P/'relkit/rdl_l117.py') for p in (P/'evidence/l117/paper').glob('*/audit.json'))
bp=P/'_budget_l117.json';b=json.loads(bp.read_text());assert sum(r['workers'] for r in b['reservations'])==6;b['status']='COMPLETE';b['recorded_worker_resource_usd']=s['worker_resource_usd'];b['billing_total']='NOT_ITEMIZED';bp.write_text(json.dumps(b,indent=2))
files=sorted(P.glob('_*l117*.py'))+[P/'relkit/rdl_l117.py',P/'requirements-l117-runtime.txt',R/'modal/l117_repro.py']+sorted((P/'sources/l117').rglob('*.py'))+[p for p in sorted((P/'evidence/l117').rglob('*')) if p.is_file()]
(P/'_sources_l117.json').write_text(json.dumps({'status':'PASS','source_sha256':sha(P/'relkit/rdl_l117.py'),'files':{str(p.relative_to(R)):sha(p) for p in files},'upstream':'sources/l117/manifest.json','checks':checks},indent=2))
ledger=P/'reproductions/execution_evidence.json';a=json.loads(ledger.read_text());a['l117']={'status':'COMPLETE_SELECTED_RELEASE_REPLAY','summary':'labs/evidence/l117/summary.json','protocol':'labs/l117-reproduction.md','checks':checks,'validation_mae':s['metrics']['val'],'test_mae':s['metrics']['test'],'historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'See reviews/l117-publication.json after live verification'};ledger.write_text(json.dumps(a,indent=2)+'\n');print('Recorded complete selected experiment and package checks')
