"""Bounded validation of every portable notebook code cell with full training enabled."""
import hashlib,json,fcntl,datetime,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
RATE=.00022572
training_image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
    .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
    .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
app=modal.App('l134-portable-training-validation')
image=training_image.add_local_file(ROOT/'labs/solutions/0134-training-at-scale.ipynb','/input/solution.ipynb')
volume=modal.Volume.from_name('l134-notebook-validation',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=900,retries=0,volumes={'/validation':volume})
def worker(expected_code_hash):
    import os,time,uuid,traceback
    start=time.perf_counter();out=Path('/validation');work=out/'work'
    assert not work.exists(),'Do not overwrite validation evidence'
    work.mkdir();os.chdir(work)
    nb=json.loads(Path('/input/solution.ipynb').read_text())
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    actual=hashlib.sha256('\n\n'.join(code).encode()).hexdigest();assert actual==expected_code_hash
    ns={'__name__':'__main__'};completed=0
    try:
        for i,source in enumerate(code):
            source=source.replace('RUN_FULL_REPRODUCTION = False','RUN_FULL_REPRODUCTION = True')
            exec(compile(source,f'notebook-cell-{i}','exec'),ns);completed+=1
        packet=json.loads((work/'l134-full/packet.json').read_text())
        assert len(packet['records'])==5
        result=dict(status='PASS',code_cells=completed,code_sha256=actual,full_gate=True,packet=packet,
                    seconds=time.perf_counter()-start,run_uuid=str(uuid.uuid4()),live_colab='NOT_CHECKED',
                    execution='All portable code cells in isolated pinned T4 Python namespace; not Jupyter frontend')
        result['resource_usd']=result['seconds']*RATE
        (out/'report.json').write_text(json.dumps(result,indent=2));return result
    except Exception:
        (out/'failure.json').write_text(json.dumps(dict(status='FAIL',completed_cells=completed,traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
    finally:volume.commit()
@app.local_entrypoint()
def main():
    nb=json.loads((ROOT/'labs/solutions/0134-training-at-scale.ipynb').read_text())
    code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    digest=hashlib.sha256(code.encode()).hexdigest()
    with (ROOT/'labs/_budget_l134.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
        assert not any(x['mode']=='notebook-validation' for x in b['reservations'])
        assert sum(x['workers'] for x in b['reservations'])+1<=b['maximum_workers']
        assert b['maximum_worker_usd']+5+b['overhead_reserve_usd']<=b['budget_usd']==10
        for path,sha in b['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
        b['reservations'].append(dict(mode='notebook-validation',workers=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=digest))
        b['source_hashes']['modal/l134_notebook_check.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        f.seek(0);json.dump(b,f,indent=2);f.truncate()
    print(worker.remote(digest))
