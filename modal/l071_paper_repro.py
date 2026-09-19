"""Run from repository root: modal run --detach modal/l071_paper_repro.py --preset closer

Public MNIST follow-up; NOT an exact paper-table reproduction. See labs/l071-reproduction.md.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-l071-vime')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('numpy','scikit-learn','torch')
       .add_local_file(ROOT/'labs/relkit/vime_l071.py','/workspace/relkit/vime_l071.py')
       .add_local_file(ROOT/'labs/_run_l071.py','/workspace/_run_l071.py'))
volume=modal.Volume.from_name('relational-l071-results',create_if_missing=True)
@app.function(image=image,gpu='T4',timeout=21600,volumes={'/results':volume})
def train(preset: str):
    import subprocess,sys,time
    if preset not in ('smoke','closer','paper'):raise ValueError(preset)
    output=f'/results/{preset}-{int(time.time())}.json'
    subprocess.run([sys.executable,'/workspace/_run_l071.py','--preset',preset,'--device','cuda','--output',output],check=True)
    volume.commit()
    return output
@app.local_entrypoint()
def main(preset: str='closer'):
    print(train.remote(preset))
