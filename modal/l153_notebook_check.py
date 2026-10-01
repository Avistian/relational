"""Pinned-environment execution of all default portable notebook cells."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l153-notebook-check')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').pip_install('nbformat==5.10.4','ipython==8.30.0'))
image=image.add_local_file(ROOT/'labs/solutions/0153-recommendation-portfolio.ipynb','/work/notebook.ipynb')
volume=modal.Volume.from_name('l153-recommendation-evidence')
@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=600,retries=0,volumes={'/evidence':volume})
def execute(attempt):
 import json,os,sys,time,hashlib,nbformat,traceback,types
 start=time.perf_counter();root=Path('/evidence')/attempt;root.mkdir(exist_ok=False);os.chdir(root)
 os.environ.update(XDG_CACHE_HOME='/evidence/cache',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 module=types.ModuleType('notebook_l153');sys.modules[module.__name__]=module;ns=module.__dict__;notebook=nbformat.read('/work/notebook.ipynb',4);code='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
 try:
  count=0
  for c in notebook.cells:
   if c.cell_type!='code':continue
   exec(compile(c.source,f'notebook-cell-{count}','exec'),ns);count+=1
  report=json.loads(Path('l153-report.json').read_text());report.update(code_sha256=hashlib.sha256(code.encode()).hexdigest(),cells=count,seconds=time.perf_counter()-start,full_gate='NOT_RUN: default portable execution only; training pilot measured separately')
  Path('execution.json').write_text(json.dumps(report,indent=2));return report
 except Exception:
  Path('failure.txt').write_text(traceback.format_exc());raise
 finally:
  Path('cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*.00026124)));volume.commit()
@app.local_entrypoint()
def main(attempt:str='notebook-retry1'):
 from l153_repro import reserve
 reserve(attempt,seconds=600);print(execute.remote(attempt))
