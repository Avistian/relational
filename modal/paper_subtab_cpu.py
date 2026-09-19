"""CPU fallback for accounts without GPU access; same complete SubTab implementation."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-subtab-paper-cpu')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.2.3','scipy==1.15.3','scikit-learn==1.7.2')
       .add_local_dir(ROOT/'labs/reproductions',remote_path='/root/reproductions',ignore=['**/*results.json','**/__pycache__/**']))
volume=modal.Volume.from_name('relational-paper-071-074',create_if_missing=True)
@app.function(image=image,cpu=8,memory=8192,timeout=14400,volumes={'/evidence':volume})
def run():
    import sys
    sys.path.insert(0,'/root/reproductions')
    from subtab import run_subtab
    result=run_subtab(output='/evidence/subtab-paper-results.json',threads=4)
    volume.commit()
    return {'seconds':result['seconds'],'verdict':result['verdict']}
@app.local_entrypoint()
def main():print(run.remote())
