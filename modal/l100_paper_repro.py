"""OAG HGT remote replay. Smoke NN by default; paper/release target CS, five seeds.
modal run --detach modal/l100_paper_repro.py --preset paper
Data and partial per-seed results persist; no historical-paper parity is implied.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l100-hgt-cs')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','dill==0.3.8')
 .add_local_file(ROOT/'labs/relkit/hgt_l093.py','/work/relkit/hgt_l093.py')
 .add_local_file(ROOT/'labs/_sources_l093.json','/work/_sources_l093.json')
 .add_local_file(ROOT/'labs/_run_l093.py','/work/_run_l093.py')
 .add_local_file(ROOT/'labs/_fetch_l093.py','/work/_fetch_l093.py'))
volume=modal.Volume.from_name('l100-hgt-evidence',create_if_missing=True)
@app.function(image=image,gpu='A10G',cpu=4,memory=65536,timeout=86400,volumes={'/evidence':volume})
def experiment(preset,seed):
 import json,subprocess,sys,uuid
 dataset='CS' if preset in ('paper','release') else 'NN'
 output='/evidence/'+preset+'-'+str(seed)+'-'+str(uuid.uuid4())[:8]+'.json'
 try:
  subprocess.run([sys.executable,'/work/_fetch_l093.py','--dataset',dataset,'--directory','/evidence/data'],check=True)
  record=json.loads(Path('/evidence/data',f'graph_{dataset}-download.json').read_text())
  subprocess.run([sys.executable,'/work/_run_l093.py','--preset',preset,'--data',f'/evidence/data/graph_{dataset}.pk','--sha256',record['sha256'],'--seeds',str(seed),'--device','cuda','--output',output],check=True)
 finally:volume.commit()
 return {'output':output,'summary':json.loads(Path(output).read_text())['summary']}
@app.local_entrypoint()
def main(preset:str='smoke'):
 if preset not in ('smoke','teaching','paper','release'):raise ValueError(preset)
 for seed in (range(5) if preset in ('paper','release') else [0]):print(experiment.remote(preset,seed))
