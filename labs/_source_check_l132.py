"""Check pinned bytes, instrumented runner operations and prepared archive identities."""
import ast,hashlib,json
from pathlib import Path
from _run_l132 import instrument_source
P=Path(__file__).resolve().parent;R=P.parent;manifest=json.loads((P/'sources/l132/manifest.json').read_text());checked=[]
for name,info in manifest['sources'].items():
 if 'sha256' not in info:continue
 assert hashlib.sha256((P/'sources/l132'/name).read_bytes()).hexdigest()==info['sha256'];checked.append(name)
# The complete original train/test ASTs survive instrumentation exactly.
for name in ['gnn_link.py','idgnn_link.py']:
 source=(P/'sources/l132'/name).read_text();changed=instrument_source(source)
 def funcs(text):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
 assert funcs(source)==funcs(changed);compile(changed,name,'exec')
b=json.loads((P/'_budget_l132.json').read_text())
for relative,digest in b['source_hashes'].items():assert hashlib.sha256((R/relative).read_bytes()).hexdigest()==digest,relative
assert abs(b['maximum_worker_usd']+b['overhead_reserve_usd']-10)<1e-9
r=dict(status='PASS',pinned_files=checked,original_train_test_ast='EXACT',budget_source_hashes=len(b['source_hashes']),historical_identity='NOT_ESTABLISHED')
(P/'_source_check_l132_results.json').write_text(json.dumps(r,indent=2));print(r)
