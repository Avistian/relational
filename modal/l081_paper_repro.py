"""Bounded remote reconstruction operator; historical result NOT_RUN.

modal run modal/l081_paper_repro.py --preset smoke
Full 50x3m budget belongs on a durable GPU host; see labs/l081-reproduction.md.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l081-qm9-reconstruction')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','numpy==2.2.6','torch-geometric==2.6.1','rdkit==2025.9.6').add_local_dir(ROOT/'labs/relkit',remote_path='/work/relkit'))
volume=modal.Volume.from_name('l081-qm9-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',timeout=86400,volumes={'/evidence':volume})
def experiment(preset):
    import sys,uuid
    sys.path.insert(0,'/work')
    from relkit.qm9_l081 import run_reconstruction
    if preset not in ('smoke','closer'):raise ValueError('Use a durable host for the full paper-budget search')
    output='/evidence/runs/'+str(uuid.uuid4())
    result=run_reconstruction('/evidence/data',output,preset,81,'cuda')
    volume.commit();return {'output':output,'result':result}
@app.local_entrypoint()
def main(preset:str='smoke'):
    import json
    print(json.dumps(experiment.remote(preset),indent=2))
