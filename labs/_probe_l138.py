"""Bounded full-archive inspection and real text-embedding timing; no model fit."""
import hashlib,json,time,urllib.request,zipfile
from pathlib import Path

def run(root):
 import numpy as np,pyarrow.parquet as pq,torch
 start=time.perf_counter();torch.set_num_threads(2);root=Path(root);source=Path(__file__).parent/'sources/l138'
 result=dict(status='RUNNING',archives={},tables={})
 def save():
  result['seconds']=time.perf_counter()-start;(root/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 for name,registry in [('db.zip','relbench__datasets__hashes.json'),('tasks/user-churn.zip','relbench__tasks__hashes.json')]:
  target=root/'cache'/name;target.parent.mkdir(parents=True,exist_ok=True)
  url='https://relbench.stanford.edu/download/rel-amazon/'+name
  h=hashlib.sha256();n=0;clock=time.perf_counter()
  with urllib.request.urlopen(url,timeout=90) as response,target.open('wb') as f:
   while chunk:=response.read(8*1024*1024):f.write(chunk);h.update(chunk);n+=len(chunk)
  expected=json.loads((source/registry).read_text())['rel-amazon/'+name]
  result['archives'][name]=dict(url=url,bytes=n,sha256=h.hexdigest(),historical_sha256=expected,historical_match=h.hexdigest()==expected,download_seconds=time.perf_counter()-clock)
  with zipfile.ZipFile(target) as z:
   result['archives'][name]['members']=[dict(name=i.filename,bytes=i.file_size) for i in z.infolist()]
   z.extractall(root/'unpacked')
  save()
 for p in (root/'unpacked').rglob('*.parquet'):
  f=pq.ParquetFile(p);result['tables'][str(p.relative_to(root))]=dict(rows=f.metadata.num_rows,columns=f.schema.names,metadata={k.decode():v.decode() for k,v in (f.schema_arrow.metadata or {}).items() if k!=b'pandas'})
 save()
 from sentence_transformers import SentenceTransformer
 spec=json.loads((source/'manifest.json').read_text())['text_model'];clock=time.perf_counter();model=SentenceTransformer(spec['model'],revision=spec['revision'],device='cuda')
 result['text_model']=dict(**spec,load_seconds=time.perf_counter()-clock)
 # Representative fixed row sample from each text field; read only that field.
 result['embedding_probes']=[]
 for p in (root/'unpacked').rglob('*.parquet'):
  f=pq.ParquetFile(p)
  for col in f.schema_arrow:
   if str(col.type) not in ('string','large_string'):continue
   # Source semantic-type inference uses unique count; report candidates, not definitive stypes.
   vals=[]
   for batch in f.iter_batches(batch_size=2048,columns=[col.name]):
    vals.extend('' if x is None else str(x) for x in batch.column(0).to_pylist())
    if len(vals)>=4096:break
   vals=vals[:4096]
   if not vals:continue
   model.encode(vals[:32],batch_size=32,show_progress_bar=False);torch.cuda.synchronize();clock=time.perf_counter()
   out=model.encode(vals,batch_size=256,show_progress_bar=False);torch.cuda.synchronize();elapsed=time.perf_counter()-clock
   result['embedding_probes'].append(dict(table=p.name,column=col.name,sample_rows=len(vals),total_rows=f.metadata.num_rows,seconds=elapsed,projected_seconds=elapsed*f.metadata.num_rows/len(vals),output_shape=list(out.shape),mean_chars=float(np.mean([len(v) for v in vals]))));save()
 result['status']='COMPLETE';save();return result
