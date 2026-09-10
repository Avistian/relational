"""Execute only the local ignored L060 teacher; preserve standard rich outputs."""
import os,time,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/l060-matplotlib');os.environ.setdefault('IPYTHONDIR','/tmp/l060-ipython')
from pathlib import Path
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parent;path=root/'solutions/0060-broad-model-comparison.ipynb'
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
report=dict(status='PASS',code_cells=count,seconds=time.perf_counter()-start,scope='Five live TODOs, full 210-record reanalysis, exact/MC permutations, fresh five-arm smoke and concrete EXIT; larger local gate off')
(root/'_execution_l060_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
