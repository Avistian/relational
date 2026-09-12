"""Execute the ignored L070 teacher and bind its fresh EXIT and exact files."""
import os,time,json,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/l070-matplotlib');os.environ.setdefault('IPYTHONDIR','/tmp/l070-ipython')
from pathlib import Path
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parent;path=root/'solutions/0070-foundation-model-checkpoint.ipynb';input_sha=hashlib.sha256(path.read_bytes()).hexdigest();operator_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
nb=nbformat.read(path,as_version=4);shell=TerminalInteractiveShell.instance();count=0;start=time.perf_counter();wait_seconds=0.
os.chdir(root)
for cell in nb.cells:
 if cell.cell_type!='code':continue
 if cell.source.startswith('# CHECK — exact task'):
  wait_start=time.perf_counter()
  while not (root/'_verify_l070_v2_results.json').exists():time.sleep(1)
  wait_seconds+=time.perf_counter()-wait_start
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
exit_path=Path(shell.user_ns['path']);assert exit_path.name=='exit-v2.json' and exit_path.exists()
report=dict(status='PASS',code_cells=count,seconds=time.perf_counter()-start,wait_seconds=wait_seconds,execution_seconds=time.perf_counter()-start-wait_seconds,notebook_path=str(path),notebook_input_sha256=input_sha,notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),notebook_code_sha256=hashlib.sha256(json.dumps([c.source for c in nb.cells if c.cell_type=='code'],sort_keys=True).encode()).hexdigest(),executor_sha256=operator_sha,exit_path=str(exit_path),exit_sha256=hashlib.sha256(exit_path.read_bytes()).hexdigest(),scope='Six live evaluation operations; corrected TabM source parity; 30 fresh selected fits with all 60 candidates; original105 archive audit; exact current/reference predictions; feature intervention and EXIT; broader gate off')
(root/'_execution_l070_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
