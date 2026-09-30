"""Pinned solution execution plus one extra full history fit, excluded from primary25."""
import importlib.util
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
RATE=.00022572
volume=modal.Volume.from_name('l148-ablation-evidence',create_if_missing=True)
cache=modal.Volume.from_name('l148-ablation-cache',create_if_missing=True)
app=modal.App('l148-notebook-check')
if modal.is_local():
 image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').pip_install('nbformat==5.10.4','nbclient==0.10.2','ipykernel==6.29.5').add_local_file(ROOT/'labs/solutions/0148-ablation-discipline.ipynb','/work/solution.ipynb'))
else:
 image=modal.Image.debian_slim()
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=600,retries=0,volumes={'/evidence':volume,'/cache':cache})
def check():
 import os,time,json,hashlib
 os.environ.update(XDG_CACHE_HOME='/cache',HF_HOME='/cache/huggingface',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 import nbformat
 from nbclient import NotebookClient
 start=time.perf_counter();out=Path('/evidence/notebook');out.mkdir(exist_ok=True)
 notebook=nbformat.read('/work/solution.ipynb',4)
 original_code='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
 definitions=[c.source for c in notebook.cells if c.cell_type=='code' and c.source.startswith('if RUN_FULL_REPRODUCTION:')]
 # Does not execute the all25fits local gate. Instead uses live notebook definitions.
 code="RUN_FULL_REPRODUCTION=True\n"+'\n'.join(definitions)+"\ng=torch.load('/evidence/prepared/graph.pt',weights_only=False)\nextra=fit_ablation(g['data'],g['stats'],g['task'],100,Path('/evidence/notebook/history-100'),10,'cuda',None,'history')\nprint(extra['scores'])\n"
 notebook.cells.append(nbformat.v4.new_code_cell(code))
 try:
  NotebookClient(notebook,timeout=1500,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
  nbformat.write(notebook,out/'executed.ipynb')
  report=dict(status='PASS',original_code_sha256=hashlib.sha256(original_code.encode()).hexdigest(),code_cells=sum(c.cell_type=='code' for c in notebook.cells),extra_fit='history seed100 full10epochs, excluded from primary',full25fit_notebook_gate='NOT_RUN',live_colab='NOT_CHECKED')
  (out/'execution.json').write_text(json.dumps(report,indent=2));return report
 finally:
  elapsed=time.perf_counter()-start;(out/'cost.json').write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*RATE)));volume.commit()
@app.local_entrypoint()
def main():
 spec=importlib.util.spec_from_file_location('l148_repro',ROOT/'modal/l148_repro.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 r.reserve(['notebook-recovery'],600);print(check.remote())
