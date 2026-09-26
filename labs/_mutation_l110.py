"""Run the notebook's own CHECKs against correct and broken live task bodies."""
import ast,json
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent;nb=nbformat.read(P/'solutions/0110-temporal-gnn-checkpoint.ipynb',as_version=4)
ns={};checked=[]
mutants={'restore_checkpoint':('weights_only',"def restore_checkpoint(model, checkpoint):\n    model.load_state_dict(checkpoint['weights'])"),'strict_batches':('split_ties',"def strict_batches(events,size=200):\n    yield from batches(events,size)"),'legal_history':('ignore_arrival',"def legal_history(event_time,observed_time,query_time):\n    return event_time<query_time")}
# Compile canonical code first so helper definition order is immaterial.
exec(compile((P/'relkit/checkpoint_l110.py').read_text(),'canonical','exec'),ns)
for cell in nb.cells:
 if cell.cell_type!='code':continue
 task=next((name for name in mutants if 'print("PASS:' in cell.source and ('check_restore('+name+')' in cell.source or 'check_batches('+name+')' in cell.source or 'check_legal('+name+')' in cell.source)),None)
 if not task:continue
 exec(cell.source,ns);good=ns[task];label,body=mutants[task];exec(body,ns)
 try:exec(cell.source,ns)
 except (AssertionError,RuntimeError):checked.append(label)
 else:raise AssertionError('Notebook CHECK accepted '+label)
 ns[task]=good
assert len(checked)==3
student=nbformat.read(P/'0110-temporal-gnn-checkpoint.ipynb',as_version=4)
assert sum('raise NotImplementedError("TODO:' in c.source for c in student.cells)==3
r={'status':'PASS','rejected_notebook_mutants':checked,'live_functions':'restore_checkpoint called by clean trainer; strict_batches by train/evaluate/frontier checks; legal_history by complete frontier audit','student_blanks':3}
(P/'_mutation_l110_results.json').write_text(json.dumps(r,indent=2));print(r)
