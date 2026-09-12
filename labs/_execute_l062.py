"""Execute only the local ignored L062 teacher; preserve standard rich outputs."""
import os,time,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/l062-matplotlib');os.environ.setdefault('IPYTHONDIR','/tmp/l062-ipython')
from pathlib import Path
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parent;path=root/'solutions/0062-tabpfn-v1.ipynb'
nb=nbformat.read(path,as_version=4);shell=TerminalInteractiveShell.instance();count=0;start=time.perf_counter()
os.chdir(root)
for cell in nb.cells:
 if cell.cell_type!='code':continue
 count+=1;print('EXECUTE',count,cell.source.splitlines()[0],flush=True)
 with capture_output() as captured:result=shell.run_cell(cell.source,store_history=True)
 if result.error_before_exec or result.error_in_exec:
  print(captured.stdout,captured.stderr);raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
 outputs=[]
 if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
 if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
 for rich in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
 cell.outputs=outputs;cell.execution_count=count
nb.metadata['execution_verification']={'engine':'IPython in-process','code_cells':count};nbformat.write(nb,path)
report=dict(status='PASS',code_cells=count,seconds=time.perf_counter()-start,scope='Five live TabPFN tasks, copied pretrained source parity, full real-data inference, shuffled labels, query batching and historical power fallback, live kernel mutations and saved EXIT; closer gate off')
(root/'_execution_l062_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
