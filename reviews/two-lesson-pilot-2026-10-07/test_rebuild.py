"""Ensure changed exercise code cannot inherit an old execution claim."""
import sys,tempfile,json
from pathlib import Path
import nbformat as n
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'labs'))
import _walkthrough_delivery as delivery
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);delivery.ROOT=root;delivery._SAVED.clear()
    for folder in ['lessons/content','labs/solutions','labs/html']:(root/folder).mkdir(parents=True,exist_ok=True)
    slug='0081-test'
    (root/'lessons/content'/f'{slug}.md').write_text('<!-- depth-walkthrough:start -->example<!-- depth-walkthrough:end -->')
    (root/'lessons'/f'{slug}.html').write_text('<html><head></head><body></body></html>')
    for folder in ['labs','labs/solutions']:
        code=n.v4.new_code_cell('print(1)',execution_count=1,outputs=[n.v4.new_output('stream',name='stdout',text='1\n')])
        n.write(n.v4.new_notebook(cells=[code]),root/folder/f'{slug}.ipynb')
    delivery.snapshot(81)
    for folder in ['labs','labs/solutions']:n.write(n.v4.new_notebook(cells=[n.v4.new_code_cell('print(2)')]),root/folder/f'{slug}.ipynb')
    try:delivery.finalize(81)
    except RuntimeError:pass
    else:raise AssertionError('Changed code inherited execution')
    delivery.finalize(81,reset_execution=True)
    for folder in ['labs','labs/solutions']:
        code=[c for c in n.read(root/folder/f'{slug}.ipynb',as_version=4).cells if c.cell_type=='code'][0]
        assert code.source=='print(2)' and code.execution_count is None and not code.outputs
print('PASS: changed-code preservation rejected; explicit reset retains no outputs')
