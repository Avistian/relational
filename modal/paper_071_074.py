"""Paper-track runner, separate from the former compact 'closer' operators.

modal run --detach modal/paper_071_074.py --method subtab
modal volume get relational-paper-071-074 subtab-results.json ./subtab-results.json
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('relational-paper-071-074')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.2.3','scipy==1.15.3','scikit-learn==1.7.2')
       .add_local_dir(ROOT/'labs/reproductions',remote_path='/root/reproductions',ignore=['**/*results.json','**/__pycache__/**']))
volume=modal.Volume.from_name('relational-paper-071-074',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=14400,volumes={'/evidence':volume})
def run(method,smoke):
    import sys
    sys.path.insert(0,'/root/reproductions')
    output=f'/evidence/{method}-'+('smoke' if smoke else 'paper')+'-results.json'
    if method=='subtab':
        from subtab import run_subtab
        result=run_subtab(smoke=smoke,device='cuda',output=output)
    elif method=='scarf':
        from scarf import run_scarf
        result=run_scarf(smoke=smoke,device='cuda',output=output)
    else:raise ValueError('Choose scarf or subtab; historical VIME and large FastText CARTE have their own environments')
    volume.commit()
    return {'output':output,'seconds':result['seconds'],'verdict':result['verdict']}
@app.local_entrypoint()
def main(method:str='subtab',smoke:bool=False):
    print(run.remote(method,smoke))
