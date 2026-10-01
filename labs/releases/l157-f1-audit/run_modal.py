"""Managed USD10 author run. Requires the account's Modal CLI credentials."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;RATE=.00022572
app=modal.App('l157-f1-contribution');volume=modal.Volume.from_name('l157-f1-contribution',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').add_local_dir(ROOT,'/release',ignore=['**/__pycache__/**','**/evidence/**','**/results/**','**/*.zip','budget.json']))

def reserve(phase,workers):
    import fcntl,json,datetime
    from cli import source_check
    source_check()
    with (ROOT/'budget.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
        if any(x['phase']==phase for x in b['reservations']):raise ValueError('Attempt already reserved')
        upper=workers*1800*RATE
        if sum(x['workers'] for x in b['reservations'])+workers>12 or sum(x['upper_usd'] for x in b['reservations'])+upper+b['overhead_reserve_usd']>10:raise ValueError('Budget exhausted')
        b['reservations'].append(dict(phase=phase,workers=workers,seconds=1800,upper_usd=upper,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':volume})
def worker(seed,lane):
    import os,sys,traceback
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/release')
    from cli import run_one
    dest=Path('/evidence')/lane/f'seed-{seed}'
    try:return run_one(seed,lane,dest)
    except Exception:
        dest.mkdir(parents=True,exist_ok=True);(dest/'failure.txt').write_text(traceback.format_exc());raise
    finally:volume.commit()

@app.local_entrypoint()
def main(phase:str='pilot'):
    import json,sys
    sys.path.insert(0,str(ROOT))
    if json.loads((ROOT/'labs/evidence/l157/preflight.json').read_text())['status']!='PASS':raise ValueError('Preflight missing')
    if phase=='pilot':jobs=[(0,'paper')]
    elif phase=='remaining':
        pilot=json.loads((ROOT/'labs/evidence/l157/paper/seed-0/completed.json').read_text())
        if pilot['status']!='COMPLETE' or pilot['seconds']*1.25+120>=1800:raise ValueError('Pilot runtime gate')
        jobs=[(s,'paper') for s in range(1,5)]+[(s,'fit_horizon') for s in range(5)]
    else:raise ValueError(phase)
    reserve(phase,len(jobs))
    for result in worker.starmap(jobs):print(result)
