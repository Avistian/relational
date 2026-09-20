"""Register actual L094 evidence without changing prior lesson verdicts."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;path=P/'reproductions/execution_evidence.json';ledger=json.loads(path.read_text())
files={}
for stem in ['paper','verify','audit','execution','portable','delivery','build']:
 p=P/f'_{stem}_l094_results.json';b=p.read_bytes();r=json.loads(b);files[p.name]={'sha256':hashlib.sha256(b).hexdigest(),'status':r['status']}
ledger['lesson_094']={'date':'2026-09-20','scope':'Survey taxonomy and full released NN Table1 statistics audit; all three printed rows arithmetic only','full_paper_parity':'NOT_ESTABLISHED','historical_comparison':'INCOMPARABLE','fresh_graphs':['NN'],'data_level_NOT_RUN':['CS','OAG'],'fresh_model_training_runs':0,'files':files,'commands':['.venv/bin/python labs/_run_l094.py','.venv/bin/python labs/_verify_l094.py','.venv/bin/python labs/_audit_l094.py','.venv/bin/python labs/_execute_l094.py --full-graph','/tmp/l094-portable-env/bin/python labs/_portable_l094.py','.venv/bin/python labs/_delivery_l094.py'],'rejected_inferences':['Matching node counts does not prove identical graph snapshots','A passing statistics audit does not establish model-result reproduction','Arithmetic checks on CS/OAG are not full graph replays','Prior L091–L093 artifacts are not freshly trained in L094'],'execution_guard':'3 GiB address space, one BLAS thread; first unbounded notebook attempt interrupted by WSL shutdown; cause not established','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
path.write_text(json.dumps(ledger,indent=2)+'\n');print('Registered L094 execution evidence')
