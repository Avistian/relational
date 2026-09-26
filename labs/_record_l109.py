"""Record the completed selected scope without modifying other lessons' evidence."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda path:json.loads((P/path).read_text())
checks={name:read(f'_{name}_l109_results.json')['status'] for name in ['check','audit','execution','delivery','provenance']}
assert all(value=='PASS' for value in checks.values())
verified=read('_verify_l109_results.json');assert verified['status']=='COMPLETE'
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text())
ledger['lesson_109']={'date':'2026-09-26','status':'COMPLETE','scope':'All rel-f1/driver-position labels from full released database; five Table 4 heuristics on validation/test; timestamped REG census and separate synthetic historical-version tests','source_commit':verified['source_commit'],'regenerated_labels':8712,'paper_cells_matched':10,'historical_source_identity':'NOT_ESTABLISHED','full_paper':'NOT_ESTABLISHED','actual_ingestion_history':'NOT_AVAILABLE','checks':checks,'database_rows':97606,'graph_cutoffs':310,'sql_relation_comparisons':4030,'sql_node_comparisons':2790,'protocol':'labs/l109-reproduction.md','artifacts':'labs/evidence/l109/','budget':{'limit_usd':10,'paid_compute_usd':0,'runtime':'local CPU'},'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE','commands':['.venv/bin/python labs/_verify_l109.py','.venv/bin/python labs/_graph_l109.py','.venv/bin/python labs/_execute_l109.py','.venv/bin/python labs/_delivery_l109.py']}
p.write_text(json.dumps(ledger,indent=2)+'\n')
(P/'_budget_l109.json').write_text(json.dumps({'limit_usd':10,'paid_compute_usd':0,'runtime':'local CPU only','individual_experiment_cutoff_seconds':600,'paid_retries':0,'note':'Zero provider charges; local hardware/energy cost not estimated. Successful run timings are in individual reports.'},indent=2)+'\n')
print('Recorded L109 selected reproduction, validation and evidence boundaries')
