"""MovieLens 1M, DCNv2 §7.1.1 binary task (NOT the TFRS ratings tutorial).

Downloads retain TLS verification. If a host certificate is broken, obtain a
verified archive separately and pass archive_path; never silently disable TLS.
"""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL='https://files.grouplens.org/datasets/movielens/ml-1m.zip'
MD5='c4d9eecfca2ab87c1945afe126590906'  # GroupLens ml-1m.zip.md5


def load_movielens(archive_path=None, cap=None, split_seed=48):
    archive=Path(archive_path) if archive_path else Path(__file__).parents[1]/'data/cache/ml-1m.zip'
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        temporary=archive.with_suffix('.download')
        urllib.request.urlretrieve(URL,temporary)
        if hashlib.md5(temporary.read_bytes()).hexdigest()!=MD5:
            raise ValueError('Downloaded archive differs from GroupLens checksum')
        temporary.replace(archive)
    raw=archive.read_bytes()
    if hashlib.md5(raw).hexdigest()!=MD5:
        raise ValueError('MovieLens archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        ratings=pd.read_csv(z.open('ml-1m/ratings.dat'),sep='::',engine='python',
                            names=['user','movie','rating','timestamp'])
        users=pd.read_csv(z.open('ml-1m/users.dat'),sep='::',engine='python',
                          names=['user','gender','age','occupation','zip'],dtype=str)
    # Preserve original rating indices as row IDs for a regenerable split.
    ratings=ratings[ratings.rating!=3].copy()
    full_rows=len(ratings)
    if cap is not None and cap < full_rows:
        selected,_=train_test_split(np.arange(full_rows),train_size=cap,random_state=17,
                                    stratify=(ratings.rating>=4))
        ratings=ratings.iloc[np.sort(selected)].copy()
    row_ids=ratings.index.to_numpy()
    ratings['user']=ratings.user.astype(str)
    frame=ratings.merge(users,on='user',how='left',validate='many_to_one',sort=False)
    assert len(frame)==len(ratings) and frame.gender.notna().all()
    y=(frame.rating.to_numpy()>=4).astype(np.int64)
    # Random 80/10/10 as in the paper; the original split seed/indices are absent.
    tr,rest=train_test_split(np.arange(len(y)),test_size=.2,random_state=split_seed)
    va,te=train_test_split(rest,test_size=.5,random_state=split_seed)
    tr,va,te=map(np.sort,(tr,va,te))
    columns=['user','movie','gender','age','occupation','zip']
    maps=[{v:i+1 for i,v in enumerate(sorted(frame.iloc[tr][col].astype(str).unique()))} for col in columns]
    xc=np.column_stack([frame[col].astype(str).map(m).fillna(0).to_numpy(dtype=np.int64)
                        for col,m in zip(columns,maps)])
    meta={'dataset':'MovieLens-1M','source':URL,'archive_sha256':hashlib.sha256(raw).hexdigest(),
          'archive_md5':MD5,'rows':len(y),'full_filtered_rows':full_rows,
          'label':'ratings 1/2 -> 0; 4/5 -> 1; drop 3', 'fields':columns,
          'split_seed':split_seed,'subsample_seed':17 if cap else None,
          'row_ids':row_ids.tolist(),'train':tr.tolist(),'valid':va.tolist(),'test':te.tolist(),
          'train_vocabularies':maps,'split_kind':'random 80/10/10; not a temporal or cold-start evaluation'}
    return dict(xn=np.empty((len(y),0),np.float32),xc=xc,y=y,cards=[len(m)+1 for m in maps],
                train=tr,valid=va,test=te,meta=meta)
