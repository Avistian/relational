"""Validate the portable full gate under the same pinned image and budget."""
import sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'labs/releases/l157-f1-audit';sys.path.insert(0,str(PACKAGE))
app=modal.App('l157-notebook-validation');volume=modal.Volume.from_name('l157-notebook-validation',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(PACKAGE/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html')).pip_install('nbformat==5.10.4','nbclient==0.10.2','ipykernel==6.29.5','jinja2==3.1.6').add_local_file(ROOT/'labs/solutions/0157-open-source-contribution.ipynb','/input/lesson.ipynb')
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/output':volume})
def validate():
 import hashlib,json,os,time,traceback,nbformat
 from nbclient import NotebookClient
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 dest=Path('/output/validation');dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter();n=nbformat.read('/input/lesson.ipynb',4)
 original='\n\n'.join(c.source for c in n.cells if c.cell_type=='code');hash=hashlib.sha256(original.encode()).hexdigest()
 for c in n.cells:
  if c.cell_type=='code':
   c.source=c.source.replace('RUN_FULL_REPRODUCTION=False','RUN_FULL_REPRODUCTION=True')
   if 'workspace.cleanup()' in c.source:c.source=c.source.replace('workspace.cleanup()',"import shutil\nshutil.copytree(PACKAGE_ROOT/'labs/evidence/l157',Path('/output/validation/fresh'))\nworkspace.cleanup()")
 try:
  NotebookClient(n,timeout=1700,kernel_name='python3',resources={'metadata':{'path':str(dest)}}).execute()
  report=json.loads((dest/'l157-fresh-report.json').read_text());assert report['contribution']['fits']==10 and report['predictions']==12590
  r=dict(status='PASS',fits=10,predictions=12590,original_code_sha256=hash,primary_mean_inclusion=False,seconds=time.perf_counter()-start)
  (dest/'execution.json').write_text(json.dumps(r,indent=2));return r
 except Exception:
  (dest/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (dest/'cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*.00022572)));nbformat.write(n,dest/'executed.ipynb');volume.commit()
@app.local_entrypoint()
def main():
 import json
 # Allocation is reserved before notebook build so its embedded budget is immutable.
 b=json.loads((PACKAGE/'budget.json').read_text());assert any(r['phase']=='notebook-retry' for r in b['reservations'])
 print(validate.remote())
