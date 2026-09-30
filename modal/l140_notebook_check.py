"""Execute the standalone notebook, optionally its full gate for one task."""
from pathlib import Path
import modal
volume=modal.Volume.from_name('l140-checkpoint-evidence')
volumes={'/evidence':volume,'/prior/amazon':modal.Volume.from_name('l138-amazon-evidence'),'/prior/trial':modal.Volume.from_name('l139-trial-evidence')}
RATE=.00033228
TIMEOUTS={'amazon':2300,'trial':600}
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l140-notebook-validation')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').pip_install('ipython==8.30.0','nbformat==5.10.4')).add_local_file(ROOT/'labs/solutions/0140-rdl-reproduction-checkpoint.ipynb','/notebook.ipynb')
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=2300,retries=0,volumes=volumes)
def check(task):
 import hashlib,json,os,time,traceback,sys,tempfile
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 from IPython.core.interactiveshell import InteractiveShell
 root=Path('/evidence');name='notebook-'+task;marker=root/(name+'-started');assert not marker.exists();marker.touch();volume.commit();start=time.perf_counter()
 try:
  notebook=json.loads(Path('/notebook.ipynb').read_text());cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'];digest=hashlib.sha256('\n\n'.join(cells).encode()).hexdigest()
  with tempfile.TemporaryDirectory() as tmp:
   os.chdir(tmp);shell=InteractiveShell.instance();shell.user_ns.pop('__file__',None)
   for i,source in enumerate(cells):
    if source.startswith('# Full fresh training is separate'):
     shell.user_ns.update(RUN_FULL_REPRODUCTION=True,FULL_TASKS=[task],FULL_SEEDS=[100],PREPARED_ROOTS={task:'/prior/'+task})
     Path('l140-full').mkdir();Path('l140-full/preflight.json').write_text((root/'preflight.json').read_text())
    result=shell.run_cell(source,store_history=False)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError(f'Notebook cell{i} failed: {result.error_before_exec or result.error_in_exec}')
   report=json.loads(Path('l140-report.json').read_text());assert report['status']=='PASS' and report['fresh_training']=='EXECUTED'
   run=Path('l140-full')/task/'seed-100';dest=root/name;dest.mkdir()
   for filename in ['result.json','predictions.npz','source_parity.json','run-notes.json']:
    if (run/filename).exists():(dest/filename).write_bytes((run/filename).read_bytes())
   result=dict(status='PASS',task=task,full_gate_seeds=[100],code_sha256=digest,code_cells=len(cells),report=report,seconds=time.perf_counter()-start,scope='Standalone default evidence audit plus one additional complete full-data fit; extra fit excluded from primary five-seed mean',live_colab='NOT_CHECKED')
   (root/(name+'.json')).write_text(json.dumps(result,indent=2));return result
 except Exception:
  (root/(name+'-error.txt')).write_text(traceback.format_exc());raise
 finally:
  os.chdir(root)  # IPython's exit hook needs a surviving working directory.
  (root/(name+'-cost.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*RATE)));volume.commit()
@app.local_entrypoint()
def main(task:str='trial',attempt:int=1):
 from l140_repro import reserve
 assert task in TIMEOUTS
 reserve('notebook-'+task+'-attempt'+str(attempt),TIMEOUTS[task],RATE)
 print(check.with_options(timeout=TIMEOUTS[task]).remote(task))
