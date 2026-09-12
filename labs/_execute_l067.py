"""Execute only the local ignored L067 teacher; preserve standard rich outputs."""
import os,time,json,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/l067-matplotlib');os.environ.setdefault('IPYTHONDIR','/tmp/l067-ipython')
from pathlib import Path
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parent;path=root/'solutions/0067-local-pfn-retrieval-finetuning.ipynb'
nb=nbformat.read(path,as_version=4);shell=TerminalInteractiveShell.instance();count=0;start=time.perf_counter()
os.environ['L067_OUTPUT']=str(root/'data/cache/l067-teacher'/str(time.time_ns()))
os.chdir(root)
for cell in nb.cells:
 if cell.cell_type!='code':continue
 if '--preflight' in __import__('sys').argv and cell.source.startswith('# PROVIDED — fresh current-kernel'):break
 count+=1;print('EXECUTE',count,cell.source.splitlines()[0],flush=True)
 with capture_output() as captured:result=shell.run_cell(cell.source,store_history=True)
 if result.error_before_exec or result.error_in_exec:
  print(captured.stdout,captured.stderr);raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
 outputs=[]
 if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
 if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
 for rich in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
 cell.outputs=outputs;cell.execution_count=count
if '--preflight' in __import__('sys').argv:
 print('Preflight passed',count,'code cells; measured experiment not run');raise SystemExit(0)
nb.metadata['execution_verification']={'engine':'IPython in-process','code_cells':count};nbformat.write(nb,path)
exit_path=Path(os.environ['L067_OUTPUT'])/'exit-v2.json'
report=dict(status='PASS',exit_path=str(exit_path),exit_sha256=hashlib.sha256(exit_path.read_bytes()).hexdigest(),notebook_path=str(path),notebook_sha256_at_execution=hashlib.sha256(path.read_bytes()).hexdigest(),executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),code_cells=count,seconds=time.perf_counter()-start,scope='Five live local adaptation tasks; full original source forward/all parameter gradients/AdamW parity; source float32 constant normalization; stale helper/default/weight/runtime/hook rejection; fresh complete diabetes adaptation, author correspondence, EXIT; broader gate off')
(root/'_execution_l067_v2_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
