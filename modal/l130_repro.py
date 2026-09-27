"""L130 fresh bounded pilot and five full F1 runs; USD10 aggregate plan."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l130-temporal-graphs')
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
    .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
    .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html')
    .add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py')
    .add_local_file(ROOT/'labs/_run_l117.py','/work/_run_l117.py')
    .add_local_file(ROOT/'labs/relkit/checkpoint_l130.py','/work/relkit/checkpoint_l130.py')
    .add_local_file(ROOT/'labs/_run_l130.py','/work/_run_l130.py')
    .add_local_file(ROOT/'labs/relkit/batch_audit_l123.py','/work/relkit/batch_audit_l123.py')
    .add_local_dir(ROOT/'labs/sources/l117','/work/sources/l117'))
volume=modal.Volume.from_name('l130-reg-evidence',create_if_missing=True)
RATE=.00022572
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=3600,retries=0,volumes={'/evidence':volume})
def worker(seed,epochs,preset):
    import sys,time,json,datetime,uuid,hashlib
    sys.path.insert(0,'/work');from _run_l130 import run
    root=Path('/evidence')/preset/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
    assert not (root/'completed.json').exists(), 'Completed run cannot be overwritten'
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    start=time.perf_counter()
    try:
        result=run(seed,epochs,root)
        done=dict(status='COMPLETE',seed=seed,epochs=epochs,scores=result['scores'],seconds=time.perf_counter()-start)
        done.update(resource_usd=done['seconds']*RATE,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_uuid=str(uuid.uuid4()),lesson=130,runner_sha256=hashlib.sha256(Path('/work/_run_l130.py').read_bytes()).hexdigest())
        (root/'completed.json').write_text(json.dumps(done,indent=2));return done
    finally:volume.commit()
@app.local_entrypoint()
def main(mode:str='pilot', budget_file:str='labs/_budget_l130.json'):
    import json,fcntl,datetime,hashlib
    path=ROOT/budget_file
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);budget=json.load(f)
        assert mode in ['pilot','paper']
        for relative,expected in budget['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative
        assert budget['rate_usd_second']==RATE and budget['maximum_worker_usd']+budget['overhead_reserve_usd']<=budget['budget_usd']
        count=1 if mode=='pilot' else 5
        assert not any(r['mode']==mode for r in budget['reservations'])
        assert sum(r['workers'] for r in budget['reservations'])+count<=8
        if mode=='paper':assert budget['pilot_approved_for_full']
        budget['reservations'].append(dict(mode=mode,workers=count,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256((ROOT/'labs/relkit/rdl_l117.py').read_bytes()).hexdigest()))
        f.seek(0);json.dump(budget,f,indent=2);f.truncate()
    if mode=='pilot':print(worker.remote(100,1,'pilot'))
    else:
        for result in worker.starmap([(s,10,'paper') for s in range(5)]):print(result)
