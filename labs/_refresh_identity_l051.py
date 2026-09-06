"""Verify changed bookkeeping cells without rerunning unchanged model training.

Only stable-identity additions are allowed to differ. Hydrate the same visible
definitions, load the already-measured result, execute changed cells, and retain
outputs only for byte-for-byte unchanged cells.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('IPYTHONDIR','/tmp/l051-ipython')
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import json
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
from _build_l051 import build,HERE,SLUG
path=HERE/'solutions'/f'{SLUG}.ipynb';old=nbformat.read(path,as_version=4);new=build(True)
old_code=[c for c in old.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in old_code)
by_source={c.source:c for c in old_code};shell=TerminalInteractiveShell.instance();ns=shell.user_ns
ns['result']=json.loads((HERE/'data/cache/l051-teacher-results.json').read_text())
count=0;changed=[]
for cell in new.cells:
    if cell.cell_type!='code':continue
    count+=1;first=cell.source.splitlines()[0]
    hydrate=(first.startswith('# TODO') or (first.startswith('# PROVIDED') and not any(s in first for s in ['trains visible','YOUR runtime','Colab/local gate'])))
    edited=cell.source not in by_source
    if edited:
        assert any(s in first for s in ['environment','stable executable identity','EXIT TICKET','Colab/local gate']),first
        changed.append(first)
    if hydrate or edited:
        with capture_output() as captured:result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout,captured.stderr);raise RuntimeError(first) from (result.error_before_exec or result.error_in_exec)
    if edited:
        cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    else:cell.outputs=by_source[cell.source].outputs
    cell.execution_count=count
new.metadata['execution_verification']={'engine':'IPython in-process','code_cells':count,'identity_refresh':changed,'training_code_unchanged':True}
nbformat.write(new,path);print('PASS: unchanged trained code preserved; executed revised identity cells:',changed)
