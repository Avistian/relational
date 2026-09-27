"""Standalone notebook and original model parity in the pinned Python3.11 stack."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l138-notebook-validation')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('IPython==8.31.0'))
image=image.add_local_file(ROOT/'labs/solutions/0138-ecommerce-amazon.ipynb','/work/lesson.ipynb').add_local_file(ROOT/'labs/sources/l138/examples__model.py','/work/original_model.py')
volume=modal.Volume.from_name('l138-notebook-validation',create_if_missing=True)
@app.function(image=image,cpu=2,memory=16384,timeout=600,retries=0,volumes={'/validation':volume})
def check(attempt):
 import json,time,hashlib,os,tempfile,traceback,importlib.util,importlib.metadata
 from IPython.display import display
 start=time.perf_counter();nb=json.loads(Path('/work/lesson.ipynb').read_text());cells=[c['source'] for c in nb['cells'] if c['cell_type']=='code'];cells=[''.join(s) for s in cells];digest=hashlib.sha256('\n\n'.join(cells).encode()).hexdigest()
 try:
  with tempfile.TemporaryDirectory() as tmp:
   os.chdir(tmp);ns={'display':display}
   for i,code in enumerate(cells):exec(compile(code,f'cell-{i}','exec'),ns)
   torch=ns['torch'];data,stats,model=ns['neural_fixture'](ns['Model'])[1:]
   spec=importlib.util.spec_from_file_location('original','/work/original_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   original=module.Model(data,stats,2,128,1,'sum','batch_norm');original.load_state_dict(model.state_dict());model.eval();original.eval();model.zero_grad();original.zero_grad()
   a=model(data,'customer');b=original(data,'customer');torch.testing.assert_close(a,b,rtol=0,atol=0)
   a.sum().backward();b.sum().backward()
   for (na,pa),(nb,pb) in zip(model.named_parameters(),original.named_parameters()):
    assert na==nb
    if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
   result=dict(status='PASS',code_cells=len(cells),code_sha256=digest,seconds=time.perf_counter()-start,report=json.loads(Path('l138-report.json').read_text()),original_model_logits_gradients='EXACT',packages={k:importlib.metadata.version(k) for k in ['torch','pytorch-frame','relbench','torch-geometric']},full_training_gate='NOT_RUN',live_colab='NOT_CHECKED')
  Path(f'/validation/result-{attempt}.json').write_text(json.dumps(result,indent=2));return result
 except Exception:
  Path(f'/validation/failure-{attempt}.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main(attempt:int=2):
 import json,fcntl,hashlib
 with (ROOT/'labs/_budget_l138.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);upper=600*(2*.0000131+16*.00000222)
  phase=f'notebook_validation-{attempt}'
  assert not any(x['phase']==phase for x in b['reservations'])
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+3<=10
  b['reservations'].append(dict(phase=phase,upper_usd=upper,timeout=600,source_sha256=hashlib.sha256((ROOT/'modal/l138_notebook_check.py').read_bytes()).hexdigest()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(check.remote(attempt))
