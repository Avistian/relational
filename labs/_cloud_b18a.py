"""Explicit budget-reserved Modal runner. No automatic retries."""
from pathlib import Path
import json,time,sys
import modal
P=Path(__file__).resolve().parent
app=modal.App('relational-b18a')
volume=modal.Volume.from_name('relational-b18a-evidence',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.7.1','numpy==2.2.6','scikit-learn==1.6.1','scipy==1.15.3','einops==0.8.1','pandas==2.2.3','huggingface-hub==0.34.4','pydantic-settings==2.10.1','psutil==7.0.0','tqdm==4.67.1')
 .add_local_file(P/'sources/b18a/taco.tar.gz','/tmp/taco.tar.gz',copy=True)
 .run_commands('pip install --no-deps /tmp/taco.tar.gz')
 .add_local_file(P/'_experiment_b18a.py','/root/_experiment_b18a.py')
 .add_local_file(P/'relkit/state_b18a.py','/root/state_b18a.py'))
@app.function(image=image,gpu='A10',cpu=2,memory=16384,timeout=1800,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def execute(phase):
    from _experiment_b18a import run
    result=run(phase)
    path=Path('/evidence')/(phase+'.json')
    if path.exists():raise RuntimeError('Refuse remote overwrite')
    path.write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    print('EVIDENCE_SAVED',phase,result['status'],flush=True)
    return {'status':result['status'],'seconds':result['seconds']}
@app.local_entrypoint()
def main(phase:str='pilot'):
    if phase not in ['pilot','matrix','updates']:raise ValueError(phase)
    folder=P/'evidence/b18a';ledger=folder/'cloud-budget.json'
    if (folder/(phase+'.json')).exists():raise RuntimeError('Evidence exists; refuse repeated paid execution')
    if phase!='pilot':
        projection=json.loads((folder/'projection.json').read_text())
        if projection['decision']!='ADMIT_COMPLETE_MATRIX_AND_UPDATES':raise RuntimeError('Pilot/projection gate closed')
    budget=json.loads(ledger.read_text()) if ledger.exists() else dict(cap_usd=10,main_cap_usd=8,cap_gpu_seconds=18000,rate_usd_second=.00036772,attempts=[])
    if budget.get('active'):raise RuntimeError('Reconcile outstanding reservation first')
    reserve=2400*budget['rate_usd_second']
    used=sum(a['conservative_usd'] for a in budget['attempts'])
    if used+reserve>8 or sum(a['wall_seconds'] for a in budget['attempts'])+2400>18000:raise RuntimeError('INCOMPLETE_BUDGET_GATE')
    budget['active']=dict(phase=phase,reserve_usd=reserve,max_remote_seconds=1800)
    ledger.write_text(json.dumps(budget,indent=2)+'\n');start=time.monotonic();result=None
    try:
        result=execute.remote(phase)
        target=folder/(phase+'.json')
        if target.exists():raise RuntimeError('Refuse overwrite of previous evidence')
        target.write_bytes(b''.join(volume.read_file('/'+phase+'.json')));print(result['status'],result['seconds'],flush=True)
    finally:
        elapsed=time.monotonic()-start
        budget['attempts'].append(dict(phase=phase,wall_seconds=elapsed,conservative_usd=elapsed*budget['rate_usd_second'],status=result['status'] if result else 'FAILED'))
        budget.pop('active');ledger.write_text(json.dumps(budget,indent=2)+'\n')
    if result['status']!='COMPLETE':raise RuntimeError(result.get('error','INCOMPLETE'))
