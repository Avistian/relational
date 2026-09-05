"""L049 authors' prepared splits, plus a separate MovieLens time-transfer probe."""
import ast
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

CACHE = Path(__file__).resolve().parents[1]/'data/cache/l049'
TABLES = {'pima':'[openml]Pima-Indians-Diabetes',
          'breast':'[kaggle]Breast Cancer Dataset',
          'banknote':'[openml]Swiss-banknote-conterfeit-detection'}
HASHES = {'train':'a310571d796aa311c37c5c790044c9e001de8c41dfaa3848c52fa0cb7a0e6c90',
          'val':'f89cbbdce63a259d8da5d88d9b2b506cd0029215d4059fb73767d582d3cb0cdd',
          'test':'8a2868586777b1d5934beff4eef3705f2b2d9bc328067739ddf2d79caa81034c'}


def finish(x,y,tr,va,te,meta,columns):
    """Order only from training labels; labels outside tr never reach this fit."""
    with threadpool_limits(limits=1):
        importance = mutual_info_classif(x[tr],y[tr],random_state=49)
    order = np.argsort(-importance,kind='stable')
    meta.update(columns=columns,order=order.tolist(),importance=importance.tolist(),
                train=tr.tolist(),valid=va.tolist(),test=te.tolist(),mi_seed=49,
                x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    return dict(x=x[:,order].astype(np.float32),y=y,train=tr,valid=va,test=te,
                importance=importance[order].astype(np.float32),meta=meta)


def paper_data(name):
    CACHE.mkdir(parents=True,exist_ok=True)
    arrays,labels,columns = [],[],None
    for split in ('train','val','test'):
        path = CACHE/f'{split}-small.parquet'
        if not path.exists():
            tmp = path.with_suffix('.download')
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/jyansir/excelformer/resolve/main/{path.name}',tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest()!=HASHES[split]: raise ValueError('Dataset release changed')
            tmp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=HASHES[split]: raise ValueError('Dataset checksum mismatch')
        df = pd.read_parquet(path)
        record = df.loc[df.dataset_name==TABLES[name]].iloc[0]
        data = ast.literal_eval(record.table)
        assert not data['X_cat'], 'Numeric-only mirror scope'
        frame = pd.DataFrame.from_dict(data['X_num']).sort_index()
        if columns is None: columns = list(frame.columns)
        assert list(frame.columns)==columns
        arrays.append(frame.to_numpy(dtype=np.float32))
        labels.append(np.asarray(data['y'],dtype=np.int64))
    x,y = np.concatenate(arrays),np.concatenate(labels)
    assert set(np.unique(y))=={0,1} and np.isfinite(x).all()
    a,b = len(arrays[0]),len(arrays[0])+len(arrays[1])
    return finish(x,y,np.arange(a),np.arange(a,b),np.arange(b,len(x)),
                  {'dataset':TABLES[name],'split':'author-released prepared splits',
                   'source':'https://huggingface.co/datasets/jyansir/excelformer',
                   'file_sha256':HASHES,'preprocessing':'Author-prepared numeric values, used unchanged; upstream fit scope not independently reconstructed.'},columns)


def temporal_data(kind='temporal',cap=12000):
    """True timestamps; same sampled events in both regimes, no fabricated time."""
    from .dcnv2_data import load_movielens, MD5
    archive = CACHE.parent/'ml-1m.zip'
    if not archive.exists(): load_movielens(cap=100)
    raw = archive.read_bytes()
    assert hashlib.md5(raw).hexdigest()==MD5
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        events=pd.read_csv(z.open('ml-1m/ratings.dat'),sep='::',engine='python',
                           names=['user','movie','rating','timestamp'])
        users=pd.read_csv(z.open('ml-1m/users.dat'),sep='::',engine='python',
                          names=['user','gender','age','occupation','zip'],dtype=str)
    events=events.loc[events.rating!=3].copy()
    events=events.iloc[np.sort(np.random.default_rng(49).choice(len(events),min(cap,len(events)),replace=False))].copy()
    row_ids=events.index.to_numpy(); events['user']=events.user.astype(str)
    frame=events.merge(users,on='user',validate='many_to_one',sort=False)
    y=(frame.rating.to_numpy()>=4).astype(np.int64); times=frame.timestamp.to_numpy()
    if kind=='temporal':
        cuts=np.quantile(times,[.64,.80],method='nearest')
        tr=np.flatnonzero(times<cuts[0]);va=np.flatnonzero((times>=cuts[0])&(times<cuts[1]));te=np.flatnonzero(times>=cuts[1])
        assert times[tr].max()<times[va].min() and times[va].max()<times[te].min()
    elif kind=='random':
        tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=49,stratify=y)
        tr,va=train_test_split(tr,test_size=.2,random_state=49,stratify=y[tr])
    else: raise ValueError(kind)
    # Frequency representation is deliberately modest and label-free. No integer-ID magnitude.
    fields=['user','movie','gender','occupation','zip']
    raw_x=[frame.age.to_numpy(dtype=np.float32)]
    for col in fields:
        counts=frame.iloc[tr][col].value_counts(normalize=True)
        raw_x.append(frame[col].map(counts).fillna(0).to_numpy(dtype=np.float32))
    raw_x=np.column_stack(raw_x)
    qt=QuantileTransformer(n_quantiles=min(100,len(tr)),output_distribution='normal',random_state=49)
    qt.fit(raw_x[tr]);x=qt.transform(raw_x).astype(np.float32)
    meta={'dataset':'MovieLens-1M binary transfer probe','split':kind,'split_seed':49,
          'sample_seed':49,'row_ids':row_ids.tolist(),'timestamp_ranges':{k:[int(times[ii].min()),int(times[ii].max())] for k,ii in [('train',tr),('valid',va),('test',te)]},
          'archive_sha256':hashlib.sha256(raw).hexdigest(),
          'preprocessing':'Training-only frequency encoding and quantile fit; timestamp/rating excluded from inputs. Snapshot demographics; availability beyond this snapshot unverified.'}
    return finish(x,y,np.sort(tr),np.sort(va),np.sort(te),meta,['age']+[f'{f}_train_frequency' for f in fields])
