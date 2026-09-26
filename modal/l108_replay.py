"""Bounded ten-checkpoint TGAT evaluation; run pilot before full scope."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l108-temporal-sampling')
image=modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1')
for name in ['relkit/tgat_l103.py','relkit/leakage_l104.py','relkit/sampling_l108.py','_run_l108.py','_inputs_l108.json']:
 image=image.add_local_file(ROOT/'labs'/name,'/work/'+name)
inputs=modal.Volume.from_name('l103-tgat-evidence');outputs=modal.Volume.from_name('l108-sampling-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=3000,retries=0,max_containers=4,volumes={'/input':inputs,'/output':outputs})
def experiment(seed,pilot=False):
 import sys,json,torch
 sys.path.insert(0,'/work')
 from _run_l108 import run,fingerprint
 torch.set_num_threads(1)
 m=json.loads(Path('/work/_inputs_l108.json').read_text());root=Path('/input')/m['implementation_sha256']/'paper'
 dest=Path('/output')/fingerprint()/('pilot' if pilot else 'full')/f'seed-{seed}'
 try:
  r=run(seed,root,'/tmp/wikipedia',dest,'cuda',pilot,240 if pilot else 2850)
  return {'directory':str(dest),'seconds':r['elapsed_seconds'],'status':r['status'],'metrics':r['metrics']}
 finally:outputs.commit()
@app.local_entrypoint()
def main(pilot:bool=False,seeds:str=''):
 chosen=list(map(int,seeds.split(','))) if seeds else ([0] if pilot else list(range(10)))
 assert len(chosen)==len(set(chosen)) and all(0<=s<10 for s in chosen)
 for row in experiment.starmap([(s,pilot) for s in chosen]):print(row,flush=True)
