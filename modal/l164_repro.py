"""Approved aggregate USD10 Griffin gate; reserve BEFORE each cloud dispatch."""
import json,fcntl,hashlib
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];RATE=.00034544
app=modal.App('l164-griffin')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install('numpy==1.26.4','datasets==3.6.0','accelerate==1.6.0','seaborn==0.13.2','torchmetrics==1.7.1','torch-geometric==2.6.1','scikit-learn==1.5.2','tensorboard==2.19.0','safetensors==0.5.3','matplotlib==3.10.1','pyyaml==6.0.2'))
image=image.add_local_dir(ROOT/'labs/sources/l164/upstream','/work/sources/l164/upstream').add_local_file(ROOT/'labs/_run_l164.py','/work/_run_l164.py').add_local_dir('/tmp/l164-release','/input',ignore=['data'],copy=False)
volume=modal.Volume.from_name('l164-griffin-evidence',create_if_missing=True)

def reserve(phase,seconds):
    p=ROOT/'labs/evidence/l164/budget.json'
    with p.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
        for name,digest in b['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Changed worker source: '+name
        assert phase not in [x['phase'] for x in b['reservations']],'Attempt already reserved'
        amount=seconds*RATE
        assert b['overhead_reserve_usd']+amount+sum(x['upper_usd'] for x in b['reservations'])<=b['cap_usd']-b['untouched_reserve_usd'],'Aggregate budget gate'
        b['reservations'].append(dict(phase=phase,seconds=seconds,upper_usd=amount))
        f.seek(0);json.dump(b,f,indent=2);f.truncate()

def execute(phase,arm,size,seed):
    import sys,os,time,traceback,shutil,yaml
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1');sys.path.insert(0,'/work')
    from _run_l164 import pilot,run_source
    root=Path('/input');data=Path('/tmp/l164-data');data.mkdir(exist_ok=True)
    # The upload keeps immutable raw metadata; recreate the complete F1 view locally.
    for name in ['metanode.yaml','metaadj.yaml','metatask.yaml']:
        d=yaml.safe_load((root/'raw'/name).read_text());d={k:v for k,v in d.items() if k.startswith('rel-f1-') and (name!='metatask.yaml' or k=='rel-f1-driver-dnf')};(data/name).write_text(yaml.safe_dump(d,sort_keys=False))
    for name in ['node','edge','task','edgenameemb.pt','featnameemb.pt','tasknameemb.pt']:(data/name).symlink_to(root/'raw'/name)
    # Use writable root for source's relative float weight paths.
    runroot=Path('/tmp/l164-root');runroot.mkdir(exist_ok=True)
    (runroot/'data').symlink_to(data);(runroot/'checkpoint').symlink_to(root/'checkpoint')
    for name in ['floatenc-512.pt','floatdec-512.pt']:(runroot/name).symlink_to(root/name)
    out=Path('/evidence')/phase;assert not out.exists(),'Immutable attempt';out.mkdir();start=time.perf_counter()
    try:return pilot(runroot,out) if phase.startswith('pilot') else run_source(runroot,out,arm,size,seed)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate_usd_per_second=RATE)));volume.commit()

@app.function(image=image,gpu='L4',cpu=(4,4),memory=(32768,32768),timeout=1500,retries=0,volumes={'/evidence':volume},scaledown_window=2)
def probe():return execute('pilot-1','others-2',512,42)

@app.function(image=image,gpu='L4',cpu=(4,4),memory=(32768,32768),timeout=3600,retries=0,volumes={'/evidence':volume},scaledown_window=2)
def fit512(arm,seed):return execute(f'{arm}-512-{seed}',arm,512,seed)

@app.function(image=image,gpu='L4',cpu=(4,4),memory=(32768,32768),timeout=16000,retries=0,volumes={'/evidence':volume},scaledown_window=2)
def fit4096(arm,seed):return execute(f'{arm}-4096-{seed}',arm,4096,seed)

@app.local_entrypoint()
def main(phase:str='pilot'):
    if phase=='pilot':
        reserve('pilot-1',1500);print(probe.remote())
    elif phase=='full':
        decision=json.loads((ROOT/'labs/evidence/l164/cost-decision.json').read_text());assert decision['decision']=='PROCEED','Full20fit budget projection failed'
        # Reserve ALL20 fits before the first dispatch; no unreserved retry.
        assert decision['reserved_seconds']==196000
        reserve('full-20',196000)
        for arm in ['no-pretrain','others-2']:
            for size in [512,4096]:
                for seed in range(42,47):print((fit512 if size==512 else fit4096).remote(arm,seed))
    else:raise ValueError(phase)
