"""Regression for the full GPU packet's missing hashed source file."""
import ast,json
from pathlib import Path
P=Path(__file__).resolve().parent;n=json.loads((P/'solutions/0155-compare-manual-fe.ipynb').read_text())
code='\n\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
node=next(x for x in ast.walk(ast.parse(code)) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in x.targets))
files=ast.literal_eval(node.value)
for name in ['_run_l152.py','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py','sources/l117/model.py','sources/l117/nn.py']:
 assert files[name]==(P/name).read_text(),name
print('PASS: hashed GPU sources included, byte-identical')
