"""Check deterministic authored artifacts while retaining the executed solution."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0185-causal-relational-data'
paths=[R/'lessons'/(S+'.html'),R/'reference/causal-relational-data.html',P/(S+'.ipynb'),P/'evidence/l185/report.md',P/'sources/l185/requirements.txt']
before={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
solution=P/'solutions'/(S+'.ipynb');executed=solution.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_l185.py')],check=True)
 after={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 assert before==after
finally:solution.write_bytes(executed)
result={'status':'PASS','deterministic_files':before,'executed_solution_restored':True}
(P/'_determinism_l185_results.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS: deterministic author build; executed solution retained')
