"""Exact full-table FastText cache; avoids reloading 7.2GB for every benchmark split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FASTTEXT_SHA256='14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca'


class SentenceVectorCache:
    def __init__(self,names,vectors):
        self.lookup={str(name):i for i,name in enumerate(names)};self.vectors=vectors
    def get_sentence_vector(self,name):
        return self.vectors[self.lookup[name]].copy()


def table_names(frame):
    """Cover both historical lowercased relations and later case-preserving relations."""
    names=set()
    for column in frame:
        normalized=str(column).replace('\n',' ')
        names.update([normalized,normalized.lower()])
        if not pd.api.types.is_numeric_dtype(frame[column]):
            names.update(str(value).replace('\n',' ').lower() for value in frame[column].dropna().unique())
    return sorted(names)


def full_table_vectors(frame,fasttext_path,cache_dir):
    names=table_names(frame);names_hash=hashlib.sha256(json.dumps(names).encode()).hexdigest()
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'full-vectors-{names_hash[:20]}.npz'
    identity=path.with_suffix('.json')
    if path.exists() and identity.exists():
        meta=json.loads(identity.read_text())
        if meta['fasttext_sha256']!=FASTTEXT_SHA256 or meta['names_sha256']!=names_hash:raise ValueError('Vector cache identity differs')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['cache_sha256']:raise ValueError('Vector cache digest differs')
        with np.load(path,allow_pickle=False) as saved:
            return SentenceVectorCache(saved['names'],saved['vectors']),meta
    print('Preparing full-table FastText vectors once; no target labels or fitted statistics are used.',flush=True)
    digest=hashlib.sha256()
    with open(fasttext_path,'rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=FASTTEXT_SHA256:raise ValueError('Expected the original English cc.en.300.bin model')
    import fasttext
    model=fasttext.load_model(str(fasttext_path))
    vectors=np.stack([model.get_sentence_vector(name) for name in names]).astype('float32')
    # Check exact reuse before releasing the language model; this is caching, not an approximation.
    cache=SentenceVectorCache(names,vectors)
    for i in np.linspace(0,len(names)-1,min(len(names),32),dtype=int):
        assert np.array_equal(cache.get_sentence_vector(names[i]),model.get_sentence_vector(names[i]))
    del model
    np.savez_compressed(path,names=np.array(names),vectors=vectors)
    meta=dict(fasttext_sha256=FASTTEXT_SHA256,names_sha256=names_hash,names=len(names),
              cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sampled_vector_parity='EXACT',sampled_names=min(len(names),32))
    identity.write_text(json.dumps(meta,indent=2))
    return cache,meta

if __name__=='__main__':
    import argparse
    from carte import fetch_carte_data
    p=argparse.ArgumentParser();p.add_argument('--fasttext',required=True);p.add_argument('--dataset',default='wine_vivino_price');a=p.parse_args()
    root=Path.home()/'.cache/relational-paper/carte'
    frame,cfg,_=fetch_carte_data(a.dataset,root)
    _,meta=full_table_vectors(frame.drop(columns=cfg['target_name']),a.fasttext,root)
    print(json.dumps(meta,indent=2))
