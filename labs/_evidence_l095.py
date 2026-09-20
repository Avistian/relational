"""Record actual evidence and rejected inferences; never promote preparation to mastery."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
path=P/'reproductions/execution_evidence.json';ledger=json.loads(path.read_text())
files=['_experiment_l095_results.json','_verify_l095_results.json','_audit_l095_results.json','_execution_l095_results.json','_portable_l095_results.json','_delivery_l095_results.json']
artifacts={}
for name in files:
    file=P/name;record=json.loads(file.read_text());artifacts[name]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'status':record['status']}
ledger['lesson_095']={'date':'2026-09-20','scope':'Full ML-100K release data contract plus complete five-fold course ranking experiment','published_release_contract':'MATCH','course_experiment':'MEASURED','full_paper_parity':'NOT_APPLICABLE_NO_MODEL_PAPER','historical_model_score_parity':'NOT_ESTABLISHED','completed_folds':5,'planned_folds':5,'artifacts':artifacts,'commands':['OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l095.py','.venv/bin/python labs/_verify_l095.py','.venv/bin/python labs/_audit_l095.py','.venv/bin/python labs/_execute_l095.py','/tmp/l095-portable-env/bin/python labs/_portable_l095.py','.venv/bin/python labs/_delivery_l095.py'],'source_sha256':hashlib.sha256((P/'relkit/bipartite_l095.py').read_bytes()).hexdigest(),'rejected_inferences':['Matching release statistics does not reproduce a historical recommendation score','Course thresholds, selection and ranking are not claimed as a published model protocol','Five folds are not five independent datasets','Offline random completion does not establish temporal forecasting performance','A local portable replay is not a live Colab run'],'learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
path.write_text(json.dumps(ledger,indent=2)+'\n');print('Recorded L095 execution evidence')
