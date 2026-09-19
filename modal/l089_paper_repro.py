"""Full PPI Table10 release reconstruction; explicit smoke/closer/paper budgets.

.venv/bin/modal run --detach modal/l089_paper_repro.py --preset paper
Artifacts persist in volume l089-cluster-gcn-evidence. No historical parity claim.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l089-cluster-gcn')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.1','pymetis==2025.2.2')
       .add_local_file(ROOT/'labs/relkit/cluster_gcn_l089.py','/work/relkit/cluster_gcn_l089.py')
       .add_local_file(ROOT/'labs/_run_l089.py','/work/_run_l089.py'))
volume=modal.Volume.from_name('l089-cluster-gcn-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=14400,volumes={'/evidence':volume})
def experiment(preset,seed):
    import json,subprocess,sys,uuid
    run='/evidence/'+preset+'-seed'+str(seed)+'-'+str(uuid.uuid4())[:8]
    try:
        subprocess.run([sys.executable,'/work/_run_l089.py','--preset',preset,'--seed',str(seed),'--device','cuda','--data','/evidence/data','--output',run],check=True)
    finally: volume.commit()
    return {'output':run,'result':json.loads(Path(run,'result.json').read_text())}
@app.local_entrypoint()
def main(preset:str='smoke',seed:int=1):
    import json
    if preset not in ('smoke','closer','paper'):raise ValueError(preset)
    r=experiment.remote(preset,seed)
    print(json.dumps({'output':r['output'],'test':r['result']['test'],'status':r['result']['status']}))
