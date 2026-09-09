"""Fetch immutable, checksum-verified historical checkpoints; no credentials needed."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def fetch(which='all'):
    manifest=json.loads((ROOT/'_sources_foundation.json').read_text())['checkpoints']
    targets={'v1':'v1/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt','v2':'tabpfn-v2.ckpt','tabicl':'tabicl-v1.1.ckpt'}
    for name in (list(targets) if which=='all' else [which]):
        item=manifest[name];path=ROOT/'data/cache/foundation'/targets[name];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix('.partial');urllib.request.urlretrieve(item['url'],partial)
            if hashlib.sha256(partial.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'{name}: checkpoint checksum mismatch')
            partial.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'{name}: cached checkpoint checksum mismatch')
        print(name,path)


def fetch_current():
    manifest=json.loads((ROOT/'_sources_foundation.json').read_text())['current_checkpoints']
    for name,item in manifest.items():
        path=ROOT/'data/cache/foundation'/item['filename'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix('.partial');urllib.request.urlretrieve(item['url'],partial)
            if hashlib.sha256(partial.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'{name}: checksum mismatch')
            partial.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'{name}: checksum mismatch')
        print(name,path)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--which',choices=['all','v1','v2','tabicl'],default='all');fetch(p.parse_args().which)
