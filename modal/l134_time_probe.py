"""Read-only timestamp-unit diagnosis; bounded CPU recovery work."""
from pathlib import Path
import modal
app=modal.App('l134-time-probe');v=modal.Volume.from_name('l134-scale-evidence')
image=modal.Image.debian_slim().pip_install('pandas==2.2.3','pyarrow==18.1.0','numpy==1.26.4')
@app.function(image=image,timeout=180,cpu=.25,memory=2048,retries=0,volumes={'/evidence':v})
def inspect():
 import zipfile,json,pyarrow.parquet as pq,shutil
 import pandas as pd
 r={}
 with zipfile.ZipFile('/evidence/attempt-1/db.zip') as z:
  for name,col in [('db/users.parquet','CreationDate'),('db/votes.parquet','CreationDate')]:
   with z.open(name) as src,open('/tmp/table.parquet','wb') as dst:shutil.copyfileobj(src,dst)
   p=pq.ParquetFile('/tmp/table.parquet');meta={k.decode():v.decode() for k,v in p.schema_arrow.metadata.items() if k in [b'pkey_col',b'time_col']};time=json.loads(meta['time_col']);df=p.read(columns=[json.loads(meta['pkey_col']),time]).to_pandas();s=df[time]
   r[name]=dict(metadata=meta,dtype=str(s.dtype),first=str(s.iloc[0]),raw_int=int(s.astype('int64').iloc[0]),unix_seconds=int(s.astype('datetime64[s]').astype('int64').iloc[0]))
 with zipfile.ZipFile('/evidence/attempt-3/user-engagement.zip') as z:
  with z.open('user-engagement/train.parquet') as src:
   t=pq.read_table(src);df=t.to_pandas();time=json.loads(t.schema.metadata[b'time_col']);s=df[time];r['task']=dict(dtype=str(s.dtype),first=str(s.iloc[0]),raw_int=int(s.astype('int64').iloc[0]),unix_seconds=int(s.astype('datetime64[s]').astype('int64').iloc[0]),columns=list(df.columns))
 return r
@app.local_entrypoint()
def main():
 import json
 r=inspect.remote();print(r);Path('labs/_time_probe_l134.json').write_text(json.dumps(r,indent=2))
