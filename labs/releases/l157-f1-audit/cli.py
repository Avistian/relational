"""Standalone entry points. No imports or paths outside this release."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LABS=ROOT/'labs';E=LABS/'evidence/l157'
sys.path.insert(0,str(LABS))
from relkit.contribution_l157 import verify_manifest,summarize_runs,review_claim

def source_check():
    return verify_manifest(ROOT,json.loads((ROOT/'experiment-manifest.json').read_text()))

def prepare():
    import urllib.request
    source_check();E.mkdir(parents=True,exist_ok=True);source=LABS/'sources/l129'
    specs=json.loads((source/'data-manifest.json').read_text())
    specs.append(dict(path='f1/driver-position/feats.sql',url='https://raw.githubusercontent.com/snap-stanford/relbench-user-study/445bb7a3b1230f49f8e5890ae81754d3e365680f/f1/driver-position/feats.sql',sha256='2b3639da1df6f4a3fc1968579accb675878b60ad3a5e37da730f43a8c7e98a22'))
    for spec in specs:
        dest=source/spec['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            raw=urllib.request.urlopen(spec['url'],timeout=120).read()
            if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Downloaded archive/source changed')
            dest.write_bytes(raw)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Cached input changed')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for script in ['_preflight_l156.py','_audit_fe_l156.py']:
        subprocess.run([sys.executable,str(LABS/script)],cwd=ROOT,env=env,timeout=3600,check=True)
    print('PASS: isolated full archive/label/SQL preflight')

def run_one(seed,lane,output):
    source_check()
    if seed not in range(5) or lane not in ['paper','fit_horizon']:raise ValueError('Protocol does not allow that run')
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','relbench':'1.1.0','pytorch-frame':'0.2.3','torch-geometric':'2.6.1','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        if md.version(name).split('+')[0]!=version:raise ValueError('Use pinned GPU runtime: '+name)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Full run requires pinned CUDA GPU environment')
    from _run_l156 import run
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    (output/'started.json').write_text(json.dumps(dict(seed=seed,lane=lane,package_manifest_sha256=hashlib.sha256((ROOT/'experiment-manifest.json').read_bytes()).hexdigest())))
    try:
        r=run(seed,10,output,'released' if lane=='paper' else lane)
        done=dict(status='COMPLETE',seed=seed,lane=lane,epochs=10,seconds=time.perf_counter()-start,scores=r['scores'],package_manifest_sha256=hashlib.sha256((ROOT/'experiment-manifest.json').read_bytes()).hexdigest())
        (output/'completed.json').write_text(json.dumps(done,indent=2));return done
    finally:
        elapsed=time.perf_counter()-start
        (output/'cost.json').write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*.00022572)))

def analyze(freeze=False):
    source_check()
    from _replay_l156 import replay,render_report
    path=E/'input-manifest.json'
    if freeze:
        if path.exists():raise ValueError('Evidence already frozen')
        names=['preflight.json','fe/sql-audit.json','label-events.npz']+[f'{s}-{kind}.npz' for s in ['train','val','test'] for kind in ['labels','dependencies']]
        names += [f'{lane}/seed-{seed}/{name}' for lane in ['paper','fit_horizon'] for seed in range(5) for name in ['result.json','predictions.npz','completed.json','started.json','cost.json','temporal-audit.json','audit-l156.json','diagnostics.json','audit.json','checkpoint-audit.json']]
        m=dict(scope='L157 fresh package-only execution; author evidence',files={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in names})
    else:m=json.loads(path.read_text())
    report=replay(E,m);rows=[]
    expected=hashlib.sha256((ROOT/'experiment-manifest.json').read_bytes()).hexdigest()
    for lane in ['paper','fit_horizon']:
        for record in report['lanes'][lane]['records']:
            done=json.loads((E/lane/f"seed-{record['seed']}"/'completed.json').read_text())
            if done['package_manifest_sha256']!=expected:raise ValueError('Runs used a different package')
            rows.append(dict(lane=lane,seed=record['seed'],epochs=10,train_queries=7453,val_rows=499,test_rows=760,status=done['status'],test_mae=record['test']))
    report['contribution']=summarize_runs(rows)
    evidence=dict(integrity='PASS',reproduction=report['contribution']['status'],availability='NOT_ESTABLISHED',public_url=None)
    report['claims']={k:review_claim(k,evidence) for k in ['selected_reproduction','historically_leak_free','public_contribution','whole_paper','upstream_bug']}
    if freeze:path.write_text(json.dumps(m,indent=2)+'\n')
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');(E/'report.md').write_text(render_report(report));(E/'run-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],fits=report['contribution']['fits'],predictions=report['predictions'],claims=report['claims']),indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','train','freeze-evidence','replay']);p.add_argument('--seed',type=int,default=0);p.add_argument('--lane',default='paper');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='check':print('PASS source files:',source_check())
    elif a.command=='train':print(run_one(a.seed,a.lane,E/a.lane/f'seed-{a.seed}'))
    else:analyze(a.command=='freeze-evidence')
