"""Freeze exact raw subsets from hash-pinned TabReD archives."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from _fetch_l055 import fetch,HASHES
from relkit.exit_l080 import TASKS,PRESETS
ROOT=Path(__file__).resolve().parent

def prepare(preset):
    source=fetch();cfg=PRESETS[preset];data={};panels={}
    for name in TASKS:
        base=source/name/name
        matrices=[np.load(base/(n+'.npy'),mmap_mode='r') for n in ['x_num','x_bin'] if (base/(n+'.npy')).exists()]
        y=np.load(base/'y.npy');times=np.load(base/'x_meta.npy',mmap_mode='r')[:,0]
        for regime in ['random','temporal']:
            key=name+'/'+regime;split=('random-' if regime=='random' else 'sliding-window-')+'0'
            parts={p:np.load(base/'splits'/split/(p+'.npy')) for p in ['train','val','test']}
            ids={p:np.sort(np.random.default_rng(550+i).choice(v,min(len(v),cfg['train_cap'] if p=='train' else cfg['eval_cap']),replace=False)) for i,(p,v) in enumerate(parts.items())}
            for p,idx in ids.items():
                data[key+'/'+p+'/x']=np.concatenate([m[idx] for m in matrices],1).astype('float32');data[key+'/'+p+'/y']=y[idx].astype('int64')
            panels[key]=dict(ids={p:v.tolist() for p,v in ids.items()},targets={p:y[v].astype(int).tolist() for p,v in ids.items()},
                full_sizes={p:len(v) for p,v in parts.items()},features=sum(m.shape[1] for m in matrices),
                omitted_categorical_features=np.load(base/'x_cat.npy',mmap_mode='r').shape[1] if (base/'x_cat.npy').exists() else 0,
                time_ranges={p:[int(times[v].min()),int(times[v].max())] for p,v in parts.items()},
                ordered_full_split=all(times[parts[a]].max()<=times[parts[b]].min() for a,b in [('train','val'),('val','test')]),
                strict_time_boundaries=all(times[parts[a]].max()<times[parts[b]].min() for a,b in [('train','val'),('val','test')]))
    folder=ROOT/'data/l080';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{preset}.npz'
    np.savez_compressed(path,**data)
    meta=dict(preset=preset,panels=panels,archives={n:HASHES[n] for n in TASKS},sampling_seeds=[550,551,552],npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source='TabReD pinned 2026 release; not original paper data parity')
    (folder/f'{preset}.json').write_text(json.dumps(meta,indent=2)+'\n');print(path)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='exam');prepare(p.parse_args().preset)
