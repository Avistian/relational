"""Run all nine frozen fresh course fits; never substitute for Mitra Table12."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from relkit.prior_b06 import make_task,train_one,predict_tasks
P=Path(__file__).resolve().parent

def run(output):
    cpath=P/'evidence/b06/course-protocol.json';config=json.loads(cpath.read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('run-*.json')):raise RuntimeError('Refuse overwriting completed runs; choose a new output directory')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tasks=[]
    for k,family in enumerate(config['eval_families']):
        for i in range(config['tasks_per_family']):
            seed=config['evaluation_seed_base']+1000*k+i;t=make_task(family,seed,config)
            tasks.append(dict(id=f'{family}-{i:02d}',family=family,seed=seed,x=t['x'].tolist(),raw_x=t['raw_x'].tolist(),y=t['y'].tolist()))
    raw=(json.dumps(tasks,sort_keys=True,indent=2)+'\n').encode();(out/'tasks.json').write_bytes(raw)
    identity=dict(protocol_sha256=hashlib.sha256(cpath.read_bytes()).hexdigest(),task_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256((P/'relkit/prior_b06.py').read_bytes()).hexdigest(),torch=torch.__version__,numpy=np.__version__,python=platform.python_version())
    for seed in config['seeds']:
        for arm in config['arms']:
            start=time.monotonic();model,training=train_one(config,arm,seed)
            records=predict_tasks(model,tasks,config)
            weight=out/f'weights-{arm}-{seed}.pt';torch.save(model.state_dict(),weight)
            r=dict(name=config['name'],arm=arm,seed=seed,identity=identity,training=training,records=records,weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),seconds=time.monotonic()-start)
            (out/f'run-{arm}-{seed}.json').write_text(json.dumps(r,indent=2)+'\n')
            print(arm,seed,training['prior_counts'],round(r['seconds'],2),'seconds',flush=True)
    print('Complete nine-fit course matrix',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(P/'evidence/b06/runs'));args=parser.parse_args();run(args.output)
