"""Regenerate compact real FastText cache; needs the 7.24 GB public binary once.
Use --repo /path/to/pinned/carte --fasttext /path/to/cc.en.300.bin.
"""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--fasttext',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'data/l074';out.mkdir(exist_ok=True,parents=True);repo=Path(a.repo)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert revision=='f54690da4cddbedd1e1a9113a312f85783d2c125'
    manifest={'revision':revision,'sampling':'deduplicate exact entity names then label-blind seed74 sample384; no cross-source entity resolution','datasets':{}}
    names=set()
    for name,target,entity in [('wine_pl','price','name'),('wine_dot_com_prices','Prices','Names'),('wine_vivino_price','Price','Name')]:
        path=repo/'carte_ai/data'/f'{name}.parquet';data=pd.read_parquet(path)
        frame=data.drop_duplicates(entity).sample(n=384,random_state=74).reset_index(names='source_row_id')
        original_ids=frame.pop('source_row_id').tolist()
        frame.to_parquet(out/f'{name}.parquet',index=False)
        X=frame.drop(columns=[target]);names.update(X.columns)
        for col in X.select_dtypes(exclude='number'):
            names.update(str(x).lower() for x in X[col].dropna())
        manifest['datasets'][name]={'target':target,'entity':entity,'source_rows':original_ids,'original_shape':list(data.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target_scale':'released transformed price; no further log transform'}
    # One binary load; only a small vocabulary cache travels with the notebook.
    import fasttext
    print('Loading real FastText binary',flush=True);ft=fasttext.load_model(a.fasttext)
    names=sorted(names);vectors=np.stack([ft.get_sentence_vector(n) for n in names])
    np.savez_compressed(out/'vectors.npz',names=np.array(names),vectors=vectors)
    del ft
    shutil.copyfile(repo/'carte_ai/data/etc/kg_pretrained.pt',out/'kg_pretrained.pt')
    manifest['fasttext']={'url':'https://huggingface.co/hi-paris/fastText/resolve/main/cc.en.300.bin','bytes':Path(a.fasttext).stat().st_size,'sha256':hashlib.file_digest(open(a.fasttext,'rb'),'sha256').hexdigest(),'vectors':len(names),'dimension':300}
    manifest['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.suffix in ['.parquet','.npz','.pt']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['fasttext'])
if __name__=='__main__':main()
