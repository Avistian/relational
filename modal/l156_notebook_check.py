"""Execute both complete notebook lanes in the approved pinned GPU runtime."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l156-notebook-check')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').pip_install('ipython==8.30.0','nbformat==5.10.4','nbclient==0.10.2','ipykernel==6.29.5'))
image=image.add_local_file(ROOT/'labs/solutions/0156-temporal-leakage-audit.ipynb','/work/notebook.ipynb');volume=modal.Volume.from_name('l156-regression-evidence')
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':volume})
def execute():
 import json,os,time,hashlib,nbformat,traceback
 from nbclient import NotebookClient
 start=time.perf_counter();root=Path('/evidence/notebook');root.mkdir(exist_ok=False);os.chdir(root)
 notebook=nbformat.read('/work/notebook.ipynb',4);original='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
 for c in notebook.cells:
  if c.cell_type=='code' and '# @colab-bootstrap' in c.source:c.source=c.source.replace('RUN_FULL_REPRODUCTION=False','RUN_FULL_REPRODUCTION=True')
 try:
  NotebookClient(notebook,timeout=1500,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
  report=json.loads(Path('l156-full-report.json').read_text());report.update(code_sha256=hashlib.sha256(original.encode()).hexdigest(),cells=sum(c.cell_type=='code' for c in notebook.cells),seconds=time.perf_counter()-start,environment='Pinned isolated GPU; not live Colab')
  Path('execution.json').write_text(json.dumps(report,indent=2));nbformat.write(notebook,root/'executed.ipynb');return report
 except Exception:
  Path('failure.txt').write_text(traceback.format_exc());nbformat.write(notebook,root/'failed.ipynb');raise
 finally:
  elapsed=time.perf_counter()-start;Path('cost.json').write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*.00022572)));volume.commit()
@app.local_entrypoint()
def main():
 import hashlib,json
 from l156_repro import reserve
 assert json.loads((ROOT/'labs/_execution_l156_results.json').read_text())['status']=='PASS'
 freeze={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['modal/l156_notebook_check.py','labs/solutions/0156-temporal-leakage-audit.ipynb','labs/_build_l156.py']}
 f=ROOT/'labs/evidence/l156/notebook-source-manifest.json';assert not f.exists();f.write_text(json.dumps(freeze,indent=2))
 reserve('notebook');print(execute.remote())
