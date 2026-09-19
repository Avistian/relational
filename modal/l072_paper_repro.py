"""L072 optional convergence follow-up; no original-paper parity claim."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-l072')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.8.0','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2')
       .add_local_dir(ROOT/'labs/relkit',remote_path='/root/relkit')
       .add_local_file(ROOT/'labs/_run_l072.py',remote_path='/root/_run_l072.py'))
volume=modal.Volume.from_name('relational-l072-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',timeout=7200,volumes={'/evidence':volume})
def train(preset):
    import json,sys,time
    sys.path.insert(0,'/root')
    from _run_l072 import config
    from relkit.contrastive_l072 import run_experiment
    r=run_experiment(**config(preset),device='cuda')
    path=Path('/evidence')/f'{preset}-{int(time.time())}.json';path.write_text(json.dumps(r,indent=2));volume.commit()
    return {'path':str(path),'verdict':r['verdict'],'summary':r['summary']}
@app.local_entrypoint()
def main(preset: str='closer'):
    print(train.remote(preset))
