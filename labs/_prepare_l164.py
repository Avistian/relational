"""Fetch only the complete F1 component of the immutable Griffin release."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
import yaml
P=Path(__file__).resolve().parent
DATA_REV='e0c54ceada75317b06f11f8dcda7aa8304fbb593'
MODEL_REV='bd8c5be5130f34e7faa31099d0bd81d95d0aa995'
CODE_REV='b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427'

def prepare(root):
    root=Path(root);raw=root/'raw';data=root/'data';ck=root/'checkpoint'
    for d in [raw,data,ck]:d.mkdir(parents=True,exist_ok=True)
    api=json.loads((P/'sources/l164/data-api.json').read_text())
    assert api['sha']==DATA_REV
    chosen=[f for f in api['siblings'] if 'rel-f1' in f['rfilename'] or f['rfilename'] in ['metanode.yaml','metaadj.yaml','metatask.yaml','edgenameemb.pt','featnameemb.pt','tasknameemb.pt']]
    def fetch(item):
        name,url,dst,expected=item
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.write_bytes(urllib.request.urlopen(url,timeout=90).read())
        b=dst.read_bytes();assert len(b)==expected if expected is not None else True
        return dict(path=str(dst.relative_to(root)),url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    items=[]
    for f in chosen:
        n=f['rfilename'];items.append((n,f'https://huggingface.co/datasets/yamboo/Griffin_datasets_joint_v65/resolve/{DATA_REV}/{n}',raw/n,f.get('size')))
    for n in ['model.safetensors','config.json']:
        items.append((n,f'https://huggingface.co/yamboo/Griffin_models/resolve/{MODEL_REV}/others-2/FULL/{n}',ck/n,None))
    for n in ['floatenc-512.pt','floatdec-512.pt']:
        items.append((n,f'https://raw.githubusercontent.com/yanxwb/Griffin/{CODE_REV}/{n}',root/n,None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:ledger=list(pool.map(fetch,items))
    # Retain unmodified global metadata in raw. The execution view selects a disconnected
    # component; all F1 nodes/relations/queries are preserved, not a row subsample.
    for n in ['metanode.yaml','metaadj.yaml']:
        meta=yaml.safe_load((raw/n).read_text());sub={k:v for k,v in meta.items() if k.startswith('rel-f1-')}
        (data/n).write_text(yaml.safe_dump(sub,sort_keys=False))
    meta=yaml.safe_load((raw/'metatask.yaml').read_text())
    (data/'metatask.yaml').write_text(yaml.safe_dump({'rel-f1-driver-dnf':meta['rel-f1-driver-dnf']},sort_keys=False))
    for d in ['node','edge','task']:
        link=data/d
        if not link.exists():link.symlink_to((raw/d).resolve(),target_is_directory=True)
    for n in ['edgenameemb.pt','featnameemb.pt','tasknameemb.pt']:
        link=data/n
        if not link.exists():link.symlink_to((raw/n).resolve())
    ledger.sort(key=lambda v:v['path'])
    result=dict(data_revision=DATA_REV,model_revision=MODEL_REV,code_revision=CODE_REV,files=ledger,task=meta['rel-f1-driver-dnf'],execution_view='complete disconnected F1 component; global metadata filtered only')
    (root/'input-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import sys
    r=prepare(sys.argv[1] if len(sys.argv)>1 else '/tmp/l164-release')
    print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2));print('files',len(r['files']),'bytes',sum(f['bytes'] for f in r['files']))
