"""Bounded L104 checkpoint evaluation. No model training, no automatic retries.
Pilot: .venv/bin/modal run modal/l104_replay.py --pilot
Full:  .venv/bin/modal run --detach modal/l104_replay.py
Each seed is capped at 3000s. At most 10 seeds plus 2 explicit retries fit the plan.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l104-temporal-leakage-audit')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1')
 .add_local_file(ROOT/'labs/relkit/tgat_l103.py','/work/relkit/tgat_l103.py')
 .add_local_file(ROOT/'labs/relkit/leakage_l104.py','/work/relkit/leakage_l104.py')
 .add_local_file(ROOT/'labs/_run_l104.py','/work/_run_l104.py')
 .add_local_file(ROOT/'labs/_inputs_l104.json','/work/_inputs_l104.json'))
inputs=modal.Volume.from_name('l103-tgat-evidence')
outputs=modal.Volume.from_name('l104-leakage-evidence',create_if_missing=True)

@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=3000,retries=0,max_containers=4,volumes={'/input':inputs,'/output':outputs})
def experiment(seed,pilot=False):
 import sys,json,torch,hashlib,time
 sys.path.insert(0,'/work')
 from _run_l104 import run
 torch.set_num_threads(1)
 manifest=json.loads(Path('/work/_inputs_l104.json').read_text())
 digest=hashlib.sha256(b''.join(Path('/work/'+f).read_bytes() for f in ['relkit/leakage_l104.py','_run_l104.py'])).hexdigest()
 checkpoint_root=Path('/input')/manifest['implementation_sha256']/'paper'
 dest=Path('/output')/digest/('pilot' if pilot else 'full')/f'seed-{seed}'
 try:
  r=run(seed,checkpoint_root,'/tmp/wikipedia',dest,device='cuda',pilot=pilot,max_seconds=240 if pilot else 2850)
  return {'directory':str(dest),'seconds':r['elapsed_seconds'],'status':r['status'],'comparisons':r['comparisons']}
 finally:outputs.commit()

@app.local_entrypoint()
def main(pilot:bool=False,seeds:str=''):
 chosen=list(map(int,seeds.split(','))) if seeds else ([0] if pilot else list(range(10)))
 assert len(chosen)==len(set(chosen)) and all(0<=s<10 for s in chosen)
 for row in experiment.starmap([(s,pilot) for s in chosen]):print(row,flush=True)
