"""Execute the teacher notebook with IPython in this process, without kernel sockets.

Each real cell is evaluated by IPython, with streams and rich displays captured
into standard notebook outputs. Abort on the first error; save only a full pass.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
os.environ.setdefault('IPYTHONDIR','/tmp/l051-ipython')
from pathlib import Path
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parent;path=root/'solutions/0051-why-trees-still-win.ipynb'
nb=nbformat.read(path,as_version=4);shell=TerminalInteractiveShell.instance()
shell.run_line_magic('matplotlib','inline')
count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1;print('EXECUTE',count,cell.source.splitlines()[0],flush=True)
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
    if result.error_before_exec or result.error_in_exec:
        print(captured.stdout,captured.stderr);raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
    outputs=[]
    if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    cell.outputs=outputs;cell.execution_count=count
nb.metadata['execution_verification']={'engine':'IPython in-process','code_cells':count}
nbformat.write(nb,path);print('PASS: executed and saved',count,'teacher code cells')
