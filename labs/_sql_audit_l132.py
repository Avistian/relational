"""Independently regenerate all query sponsor sets from two archived association tables."""
import hashlib,json,sqlite3,zipfile,io
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent;D=P/'results/l132/raw';con=sqlite3.connect(':memory:')
# Recover only required members from the complete hash-verified original archive if absent.
if not all((D/(n+'.parquet')).exists() for n in ['conditions_studies','sponsors_studies']):
    import tempfile,urllib.request
    D.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'db.zip';digest=hashlib.sha256()
        with urllib.request.urlopen('https://relbench.stanford.edu/download/rel-trial/db.zip',timeout=90) as response,archive.open('wb') as out:
            while chunk:=response.read(1024*1024):out.write(chunk);digest.update(chunk)
        assert digest.hexdigest()=='9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a'
        with zipfile.ZipFile(archive) as source:
            for n in ['conditions_studies','sponsors_studies']:(D/(n+'.parquet')).write_bytes(source.read('db/'+n+'.parquet'))
expected={'conditions_studies':'75698f9ec4a974f014908ed081bb82182f83dd2cc641460ef63ff3dc79ad566f','sponsors_studies':'5b1dc4cadc7b0116221a0200f6e18e5a28e4b48340028dfcdbd4782c9684ec1b'}
for name,digest in expected.items():assert hashlib.sha256((D/(name+'.parquet')).read_bytes()).hexdigest()==digest

for name in ['conditions_studies','sponsors_studies']:
 df=pd.read_parquet(D/(name+'.parquet'));df['day']=df.date.astype('int64')//1000000000
 cols=['nct_id','condition_id','day'] if name=='conditions_studies' else ['nct_id','sponsor_id']
 df[cols].to_sql(name,con,index=False);con.execute(f'CREATE INDEX idx_{name} ON {name}(nct_id)')
con.execute('CREATE INDEX idx_condition_date ON conditions_studies(day)')
result={};z=zipfile.ZipFile(P/'sources/l132/condition-sponsor-run.zip')
for split in ['train','val','test']:
 table=pd.read_parquet(io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith('/'+split+'.parquet')))))
 checked=0;pairs=0
 for date,part in table.groupby('timestamp'):
  cutoff=int(date.timestamp());rows=con.execute('SELECT DISTINCT c.condition_id,s.sponsor_id FROM conditions_studies c JOIN sponsors_studies s ON c.nct_id=s.nct_id WHERE c.day>? AND c.day<=? AND c.condition_id<3973 AND s.sponsor_id<53241',(cutoff,cutoff+365*86400)).fetchall()
  truth={}
  for src,dst in rows:truth.setdefault(src,set()).add(dst)
  archived={int(row.condition_id):set(row.sponsor_id) for row in part.itertuples()}
  assert truth==archived,(split,str(date),len(truth),len(archived))
  checked+=len(truth);pairs+=len(rows)
 result[split]=dict(queries=checked,positive_pairs=pairs)
r=dict(status='PASS',splits=result,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.parquet')},archive_origin='Extracted from hash-verified historical db.zip in cloud preparation',query_rule='condition association date in (cutoff,cutoff+365days], join sponsors by study, deduplicate, filter entity IDs to test-cap database',arrival_histories='UNAVAILABLE')
(P/'_sql_audit_l132_results.json').write_text(json.dumps(r,indent=2));print(r)
