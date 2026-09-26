"""Bind final canonical code, pinned sources, full evidence and notebook execution."""
import hashlib,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
verified=json.loads((P/'_verify_l109_results.json').read_text())
for rel,digest in verified['hashes'].items():assert sha(P/rel)==digest,rel
manifest=json.loads((P/'sources/l109/manifest.json').read_text())
for name,meta in manifest.items():assert sha(P/'sources/l109'/name)==meta['sha256'],name
nb=nbformat.read(P/'solutions/0109-database-timestamp-contracts.ipynb',as_version=4)
code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code')
execution=json.loads((P/'_execution_l109_results.json').read_text())
assert hashlib.sha256(code.encode()).hexdigest()==execution['executed_code_sha256']
for key in ['labels','results','archives']:
 assert execution['fresh'][key]==verified[key],key
for name in ['check','audit','delivery','execution']:
 assert json.loads((P/f'_{name}_l109_results.json').read_text())['status']=='PASS'
graph=json.loads((P/'evidence/l109/graph-census.json').read_text());assert graph['status']=='PASS' and graph['relation_comparisons']==4030
paths=[*sorted((P/'evidence/l109').glob('*')),*sorted((P/'sources/l109').glob('*')),P/'relkit/database_l109.py',P/'relkit/f1_l109.py',P/'_verify_l109.py',P/'_graph_l109.py',P/'_audit_l109.py',P/'_check_l109.py',P/'_build_l109.py',P/'_execute_l109.py',P/'_delivery_l109.py',P/'solutions/0109-database-timestamp-contracts.ipynb',P/'0109-database-timestamp-contracts.ipynb']
report={'status':'PASS','labels':8712,'paper_cells_matched':10,'sql_graph_relations':4030,'sql_node_tables':2790,'fresh_notebook_results':'MATCH','source_manifest_files':len(manifest),'hashes':{str(p.relative_to(P)):sha(p) for p in paths if p.is_file()},'full_paper':'NOT_ESTABLISHED','historical_identity':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
(P/'_provenance_l109_results.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='hashes'})
