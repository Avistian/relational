"""Execute all standalone notebook code with full FE gate in pinned CPU runtime."""
import hashlib,json,os,tempfile,time
from pathlib import Path
P=Path(__file__).resolve().parent;notebook=json.loads((P/'solutions/0137-error-analysis-reg.ipynb').read_text());code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'];start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='l137-fe-notebook-') as tmp:
 os.chdir(tmp);ns={'__name__':'__main__'}
 for i,s in enumerate(code):exec(compile(s.replace('RUN_FULL_FE_REPRODUCTION = False','RUN_FULL_FE_REPRODUCTION = True'),f'cell-{i}','exec'),ns)
 packet=json.loads(Path('l137-fe-full/packet.json').read_text());assert packet['trials']==50 and packet['fits']==5
 import numpy as np
 count=0;dest=P/'evidence/l137/fe-notebook';dest.mkdir(exist_ok=True)
 for seed in range(5):
  a=np.load(f'l137-fe-full/seed-{seed}.npz');ref=np.load(P/f'evidence/l137/fe/paper/seed-{seed}/predictions.npz')
  for split in ['val','test']:
   for field in ['entity','time','target']:np.testing.assert_array_equal(a[split+'_'+field],ref[split+'_'+field])
   np.testing.assert_allclose(a[split+'_pred'],ref[split+'_pred'],rtol=0,atol=1e-10);count+=len(a[split+'_pred'])
  for ext in ['npz','json']:(dest/f'seed-{seed}.{ext}').write_bytes(Path(f'l137-fe-full/seed-{seed}.{ext}').read_bytes())
 r=dict(status='PASS',code_sha256=hashlib.sha256('\n\n'.join(code).encode()).hexdigest(),code_cells=len(code),full_trials=50,fits=5,independently_compared_predictions=count,seconds=time.perf_counter()-start,cloud_usd=0,live_colab='NOT_CHECKED')
 (P/'_notebook_fe_l137_results.json').write_text(json.dumps(r,indent=2));print(r)
