"""Fetch exact weights and authenticated prepared tasks; never dispatch compute."""
def fetch176(destination):
    import hashlib,json,shutil,urllib.request
    from pathlib import Path
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((P/'evidence/l176/input-manifest.json').read_text());code=manifest['code_revision'];rev=manifest['tabicl_revision']
    urls={'RDBPFN.pt':f'https://raw.githubusercontent.com/MuLabPKU/RDBPFN/{code}/model_pretrain/checkpoints/RDBPFN/model_eval00528.pt',
          'RDBPFN_single.pt':f'https://raw.githubusercontent.com/MuLabPKU/RDBPFN/{code}/model_pretrain/checkpoints/RDBPFN_single/model_eval00360.pt',
          'tabicl-classifier-v1.1-0506.ckpt':f'https://huggingface.co/jingang/TabICL-clf/resolve/{rev}/tabicl-classifier-v1.1-0506.ckpt'}
    for name,item in manifest['files'].items():
        dest=root/name
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()!=item['sha256']:
            if name.endswith('.npz'):shutil.copyfile(P/'evidence/l176'/name,dest)
            else:
                temp=dest.with_suffix('.download')
                with urllib.request.urlopen(urls[name],timeout=120) as response,temp.open('wb') as f:shutil.copyfileobj(response,f)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Download hash mismatch')
                temp.replace(dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Input hash mismatch')
    shutil.copyfile(P/'evidence/l176/input-manifest.json',root/'input-manifest.json');return str(root)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(fetch176(a.out))
