"""Final evidence consistency and package contract; does not retrain models."""
import hashlib,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent;R=P.parent;O=P/'evidence/l129'
reports={}
for key,file in [('mechanisms','_check_l129_results.json'),('mutations','_mutation_l129_results.json'),('source','_source_check_l129_results.json'),('row_order','_order_check_l129_results.json'),('notebook','_execution_l129_results.json'),('full_gate','_gate_l129_results.json'),('delivery','_delivery_l129_results.json')]:
 r=json.loads((P/file).read_text());assert r['status']=='PASS',(key,r);reports[key]='PASS'
summary=json.loads((O/'summary.json').read_text());assert len(summary['seeds'])==5 and summary['predictions']==6295
assert json.loads((O/'sql-audit.json').read_text())['total_values']==443552
for rel,h in json.loads((P/'_sources_l129.json').read_text())['files'].items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==h,rel
for seed in range(5):
 folder=O/f'paper/seed-{seed}';r=json.loads((folder/'result.json').read_text());assert len(r['trace'])==10
 for rel,h in r['files'].items():assert hashlib.sha256((folder/rel).read_bytes()).hexdigest()==h
 for rel,h in r['source_sha256'].items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==h
 assert hashlib.sha256((O/'matrices.npz').read_bytes()).hexdigest()==r['matrix_sha256']
nb=nbformat.read(P/'solutions/0129-manual-feature-engineering.ipynb',as_version=4)
gate=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('RUN_FULL_REPRODUCTION'))
assert hashlib.sha256(gate.encode()).hexdigest()==json.loads((P/'_gate_l129_results.json').read_text())['gate_code_sha256']
manifest=json.loads((R/'lessons/manifest.json').read_text());assert sum(x['id']==129 for x in manifest['lessons'])==1
s=(R/'lessons/0129-manual-feature-engineering.html').read_text()
for value in ['3.948917','0.070469','NOT_ESTABLISHED','PENDING_WRITTEN_DEFENSE','_gate_l129_results.json']:assert value in s,value
assert '[[' not in s
r=dict(status='PASS',reports=reports,selected_experiment=summary['status'],historical_identity='NOT_ESTABLISHED',human_study='NOT_RUN',predictions=6295,sql_values=443552,full_notebook_gate='EXACT',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l129_results.json').write_text(json.dumps(r,indent=2));print(r)
