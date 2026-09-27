"""Independent SQL census, archived query identity and pinned-source bytes."""
import hashlib,io,json,sqlite3,urllib.request,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
P=Path(__file__).resolve().parent
commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639';base=f'https://raw.githubusercontent.com/snap-stanford/relbench/{commit}/'
source_map={'gnn_node.py':'examples/gnn_node.py','model.py':'examples/model.py','text_embedder.py':'examples/text_embedder.py','nn.py':'relbench/modeling/nn.py','graph.py':'relbench/modeling/graph.py','utils.py':'relbench/modeling/utils.py','LICENSE':'LICENSE'}
manifest={}
for name,remote in source_map.items():
 raw=urllib.request.urlopen(base+remote,timeout=30).read();local=P/'sources/l117'/name
 assert raw==local.read_bytes(),name
 manifest[name]={'url':base+remote,'sha256':hashlib.sha256(raw).hexdigest()}

def download(name,digest):
 b=urllib.request.urlopen('https://relbench.stanford.edu/download/rel-f1/'+name,timeout=30).read();assert hashlib.sha256(b).hexdigest()==digest
 return zipfile.ZipFile(io.BytesIO(b))
audit=json.loads((P/'evidence/l130/paper/seed-0/audit.json').read_text());tables={};meta={}
with download('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482') as z:
 for name in audit['rows']:
  raw=pq.read_table(io.BytesIO(z.read('db/'+name+'.parquet')),use_threads=False);m=raw.schema.metadata;df=raw.to_pandas();time=json.loads(m[b'time_col']);pk=json.loads(m[b'pkey_col']);fk=json.loads(m[b'fkey_col_to_pkey_table'])
  if time is not None:df=df[df[time]<=pd.Timestamp('2010-01-01')]
  assert len(df)==audit['rows'][name];tables[name]=df;meta[name]=(pk,fk)
# Independent relational JOIN counts. SQL excludes nulls and dangling references.
connection=sqlite3.connect(':memory:')
for name,df in tables.items():
 pk,fks=meta[name];cols=list(dict.fromkeys(([pk] if pk else [])+list(fks)))
 df[cols].to_sql(name,connection,index=False)
relations={}
for table,(_,fks) in meta.items():
 for fk,dst in fks.items():
  pk=meta[dst][0]
  n=connection.execute(f'SELECT COUNT(*) FROM "{table}" a JOIN "{dst}" b ON a."{fk}"=b."{pk}"').fetchone()[0]
  key='|'.join((table,'f2p_'+fk,dst));rev='|'.join((dst,'rev_f2p_'+fk,table));assert n==audit['edges'][key]==audit['edges'][rev];relations[key]=n
connection.close()
with download('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e') as z:
 for split in ['val','test']:
  df=pq.read_table(io.BytesIO(z.read('driver-position/'+split+'.parquet')),use_threads=False).to_pandas()
  for seed in range(5):
   x=np.load(P/f'evidence/l130/paper/seed-{seed}/predictions.npz')
   for key,column in [('target','position'),('entity','driverId')]:np.testing.assert_array_equal(x[split+'_'+key],df[column].to_numpy())
   np.testing.assert_array_equal(x[split+'_time'],df.date.astype('int64').to_numpy())
r={'status':'PASS','rows':sum(audit['rows'].values()),'sql_forward_relations':relations,'all_five_seed_query_targets_ids_times':'EXACT_RELEASE_ARCHIVE','upstream_files':len(manifest)}
(P/'_audit_l130_results.json').write_text(json.dumps(r,indent=2))
(P/'_upstream_l130.json').write_text(json.dumps({'commit':commit,'sources':manifest,'fey_paper_url':'https://proceedings.mlr.press/v235/fey24a.html','benchmark':'https://arxiv.org/html/2407.20060v1'},indent=2));print(r)
