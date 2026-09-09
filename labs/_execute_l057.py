"""Execute solution cells in-process with IPython, then render the blank student lab."""
import os, time, json
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from bs4 import BeautifulSoup
from urllib.parse import urlsplit,unquote
from _build_l057 import ROOT,SLUG

os.chdir(ROOT)
path=ROOT/'solutions'/(SLUG+'.ipynb')
nb=nbf.read(path,as_version=4)
shell=InteractiveShell.instance()
start=time.perf_counter();count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
    error=result.error_before_exec or result.error_in_exec
    if error:raise RuntimeError(f'Code cell {count} failed: {error}') from error
    cell.execution_count=count;cell.outputs=[]
    if captured.stdout:cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
    for out in captured.outputs:cell.outputs.append(nbf.v4.new_output('display_data',data=out.data,metadata=out.metadata))
nbf.write(nb,path)
student=nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4)
body,_=HTMLExporter(template_name='lab').from_notebook_node(student)
soup=BeautifulSoup(body,'html.parser')
for el in soup.find_all(id=True):el['id']=unquote(el['id'])
for a in soup.find_all('a',href=True):
    u=urlsplit(a['href'])
    if not u.scheme and not u.netloc and u.path:a['href']='../'+a['href']
(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))
status=dict(solution_code_cells=count,seconds=time.perf_counter()-start,runner='IPython in-process; all notebook cells; not live Colab',student_html='rendered blank student notebook')
(ROOT/'_execution_l057_results.json').write_text(json.dumps(status,indent=2)+'\n')
print(status)
