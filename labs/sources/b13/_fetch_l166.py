"""Fetch exact approved inference inputs from pinned upstream URLs; no compute dispatch."""
import argparse,hashlib,json,shutil,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def fetch166(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((P/'evidence/l166/input-manifest.json').read_text())
    code=manifest['code_revision'];rev=manifest['tabicl_revision']
    urls={
      'RDBPFN.pt':f'https://raw.githubusercontent.com/MuLabPKU/RDBPFN/{code}/model_pretrain/checkpoints/RDBPFN/model_eval00528.pt',
      'RDBPFN_single.pt':f'https://raw.githubusercontent.com/MuLabPKU/RDBPFN/{code}/model_pretrain/checkpoints/RDBPFN_single/model_eval00360.pt',
      'tabicl-classifier-v1.1-0506.ckpt':f'https://huggingface.co/jingang/TabICL-clf/resolve/{rev}/tabicl-classifier-v1.1-0506.ckpt'}
    for name,item in manifest['files'].items():
        dest=root/name
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()!=item['sha256']:
            if name=='prepared.npz':shutil.copyfile(P/'evidence/l166/prepared.npz',dest)
            else:
                tmp=dest.with_suffix('.download')
                with urllib.request.urlopen(urls[name],timeout=120) as response,tmp.open('wb') as f:shutil.copyfileobj(response,f)
                assert hashlib.sha256(tmp.read_bytes()).hexdigest()==item['sha256'],name
                tmp.replace(dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']
    shutil.copyfile(P/'evidence/l166/input-manifest.json',root/'input-manifest.json')
    return str(root)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='/tmp/l166-input');args=parser.parse_args();print(fetch166(args.out))
