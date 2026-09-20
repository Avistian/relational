"""Link executed evidence and reject stronger claims explicitly."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
path=P/'reproductions/execution_evidence.json';ledger=json.loads(path.read_text())
files=['_experiment_l096_results.json','_verify_l096_results.json','_execution_l096_results.json','_portable_l096_results.json','_delivery_l096_results.json']
artifacts={}
for name in files:
 f=P/name;r=json.loads(f.read_text());assert r['status']=='PASS',name
 artifacts[name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'status':r['status']}
ledger['lesson_096']={'date':'2026-09-20','scope':'Complete declared four-table SQL-to-graph semantic experiment','course_experiment':'PASS','full_paper_parity':'NOT_APPLICABLE_NO_MODEL_PAPER','historical_predictive_score':'NOT_ESTABLISHED','complete_databases':33,'planned_databases':33,'artifacts':artifacts,'commands':['.venv/bin/python labs/_verify_l096.py','.venv/bin/python labs/_run_l096.py','.venv/bin/python labs/_execute_l096.py','/tmp/l095-portable-env/bin/python labs/_portable_l096.py','.venv/bin/python labs/_delivery_l096.py'],'source_sha256':hashlib.sha256((P/'relkit/schema_l096.py').read_bytes()).hexdigest(),'rejected_inferences':['A SQL semantic audit is not Fey et al. predictive-result reproduction','Finite integer-key fixtures do not establish arbitrary SQL schema equivalence','Static referential integrity does not prove temporal availability','Local portable execution does not prove live Colab','Prepared materials do not establish learner mastery'],'learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
path.write_text(json.dumps(ledger,indent=2)+'\n');print('Recorded L096 evidence')
