"""Train the notebook's live code through its gated operator at smoke scale."""
import ast,contextlib,io,json
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parent
from _build_l051 import build
nb=build(True)
ns={'__name__':'l051_live_gate_check'}
for cell in nb.cells:
    if cell.cell_type!='code':continue
    first=cell.source.splitlines()[0]
    if first.startswith('# PROVIDED') and not any(x in first for x in ['trains visible','YOUR runtime','Colab/local gate']):
        exec(compile(cell.source,'<notebook provided>','exec'),ns)
    elif first.startswith('# TODO'):
        exec(compile(cell.source,'<notebook student implementation>','exec'),ns)
    elif first.startswith('# EXIT TICKET'):
        tree=ast.parse(cell.source);assign=next(n for n in tree.body if isinstance(n,ast.Assign))
        exec(compile(ast.Module(body=[assign],type_ignores=[]),'<tracked names>','exec'),ns)
calls=[];fit=ns['fit_arm']
def observed_fit(*args,**kwargs):
    calls.append(args[0]);return fit(*args,**kwargs)
ns['fit_arm']=observed_fit
gate=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('# PROVIDED — Colab/local gate'))
gate=gate.replace('RUN_PAPER_REPRO=False','RUN_PAPER_REPRO=True').replace("PAPER_PRESET='closer'","PAPER_PRESET='smoke'").replace('l051-student-closer','l051-student-smoke-v2')
out=root/'data/cache/l051-student-smoke-v2'
fresh=not (out/'summary.json').exists()
with contextlib.redirect_stdout(io.StringIO()) as capture:
    exec(compile(gate,'<live notebook gate>','exec'),ns)
assert len(calls)==(45 if fresh else 0),len(calls)
summary=json.loads((out/'summary.json').read_text());assert summary['verdict']=='INCOMPARABLE'
assert len(summary['datasets'])==3
# Same notebook settings and implementation resume without fitting another model.
calls.clear()
try:
    with contextlib.redirect_stdout(io.StringIO()) as again:exec(compile(gate,'<live notebook gate>','exec'),ns)
except ValueError:
    previous=json.loads((out/'contract.json').read_text())['identity']
    print({k:(previous.get(k),v) for k,v in ns['live_identity'].items() if previous.get(k)!=v})
    raise
assert not calls and again.getvalue().count('RESUME')==3
report=dict(live_notebook_gate='PASS',fits_on_first_run=45 if fresh else 0,resumed_datasets=3,verdict=summary['verdict'])
(root/'_gate_l051_results.json').write_text(json.dumps(report,indent=2));print(report)
