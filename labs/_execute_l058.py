"""Execute every local L058 teacher cell, including 10,000-candidate search."""
import json,os,time
from pathlib import Path
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from _build_l058 import ROOT,SLUG,render_preview
os.chdir(ROOT);path=ROOT/'solutions'/(SLUG+'.ipynb');nb=nbf.read(path,as_version=4)
shell=InteractiveShell.instance();start=time.perf_counter();count=0
for cell in nb.cells:
 if cell.cell_type!='code':continue
 count+=1
 with capture_output() as cap:r=shell.run_cell(cell.source,store_history=True)
 error=r.error_before_exec or r.error_in_exec
 if error:raise RuntimeError(f'Cell{count}: {error}') from error
 cell.execution_count=count;cell.outputs=[]
 if cap.stdout:cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=cap.stdout))
 if cap.stderr:cell.outputs.append(nbf.v4.new_output('stream',name='stderr',text=cap.stderr))
 for out in cap.outputs:cell.outputs.append(nbf.v4.new_output('display_data',data=out.data,metadata=out.metadata))
 print('Cell',count,'PASS',flush=True)
nbf.write(nb,path);render_preview()
result={'status':'PASS','code_cells':count,'seconds':time.perf_counter()-start,'runner':'IPython in-process, all cells',
 'tracks':['300-row complete rank audit','276-row missingness-aware tiny audit','1000 and 10000 proposals through live TODOs'],
 'live_calls':shell.user_ns['call_counts'],'paper_results':'INCOMPARABLE','live_colab':'NOT_CHECKED'}
(ROOT/'_execution_l058_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
