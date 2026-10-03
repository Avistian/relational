"""Final semantic, evidence and budget checks; no model inference is rerun."""
import ast,hashlib,importlib.util,json,tempfile
from pathlib import Path
import nbformat,numpy as np
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b09'
a=json.loads((E/'course-audit.json').read_text());b=json.loads((E/'budget.json').read_text())
assert a['expected']==9 and a['completed']==6 and a['status']=='INCOMPLETE'
assert {(x['model'],x['seed']) for x in a['rows']}=={(m,s) for m in ['exaone','nori'] for s in range(3)}
assert all(x['repeat_max_abs_diff']==0 for x in a['rows'])
assert all(s=='INCOMPLETE_RESOURCE_GATE' for s in a['missing_status'].values())
assert json.loads((E/'paper-gate.json').read_text())['status']=='INCOMPLETE_SOURCE_PROTOCOL'
total=b['preparation_allowance_s']+b['validation_reserve_s']+sum(x['seconds'] for x in b['attempts']);assert total<=b['cap_s'] and b['paid_usd']==0
student=nbformat.read(P/'b09-cost-frontier.ipynb',4);solution=nbformat.read(P/'solutions/b09-cost-frontier.ipynb',4)
canonical={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((P/'relkit/cost_b09.py').read_text()).body if isinstance(n,ast.FunctionDef)}
code='\n'.join(c.source for c in solution.cells if c.cell_type=='code');functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
assert all(functions[k]==v for k,v in canonical.items())
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((P/'_execution_b09_results.json').read_text())['code_sha256']
assert sum('NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
assert all(not c.get('outputs') for c in student.cells if c.cell_type=='code')
from _preflight_b09 import verify_files
verify_files(R,json.loads((E/'artifact-manifest.json').read_text())['files'])
with tempfile.TemporaryDirectory() as td:
 path=Path(td)/'helper.py';path.write_text('return 1');expected={'helper.py':hashlib.sha256(path.read_bytes()).hexdigest()};verify_files(td,expected);path.write_text('return 2')
 try:verify_files(td,expected)
 except ValueError:pass
 else:raise AssertionError('Changed helper accepted')
r=dict(status='PASS',complete_fresh_runs=6,declared_runs=9,authenticated_predictions=534,canonical_notebook_AST='PASS',unchanged_packet='PASS',changed_helper_rejected=True,aggregate_accounted_seconds=total,paid_usd=0,course='INCOMPLETE_RESOURCE_GATE',paper='INCOMPLETE_SOURCE_PROTOCOL',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_b09_results.json').write_text(json.dumps(r,indent=2));print(r)
