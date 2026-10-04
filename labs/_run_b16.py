"""Run and save the complete finite lane; never overwrite a different receipt."""
import hashlib,json,time
from pathlib import Path
from relkit.partitions_b16 import run_experiment
P=Path(__file__).resolve().parent
start=time.monotonic(); report=run_experiment()
encoded=json.dumps(report,indent=2,sort_keys=True)+'\n'
target=P/'evidence/b16/diagnostic.json'
if target.exists() and target.read_text()!=encoded:raise SystemExit('Immutable evidence differs; investigate before replacing')
target.write_text(encoded)
print(json.dumps(dict(status='PASS',subsets=len(report['subsets']),selections=len(report['selections']),graph_checks=len(report['graph_checks']),test_interventions=len(report['test_interventions']),seconds=time.monotonic()-start,sha256=hashlib.sha256(encoded.encode()).hexdigest()),indent=2))
