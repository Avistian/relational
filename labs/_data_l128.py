"""Reconstruct every released DNF label from raw rows in independent SQLite."""
import hashlib,io,json,sqlite3,urllib.request,zipfile
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
P=Path(__file__).resolve().parent
raw=urllib.request.urlopen('https://relbench.stanford.edu/download/rel-f1/db.zip',timeout=60).read()
assert hashlib.sha256(raw).hexdigest()=='ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'
z=zipfile.ZipFile(io.BytesIO(raw));df=pq.read_table(io.BytesIO(z.read('db/results.parquet')),use_threads=False).to_pandas()
df['day']=df.date.astype('int64')//1000000000
con=sqlite3.connect(':memory:');df[['driverId','statusId','day']].to_sql('results',con,index=False);con.execute('CREATE INDEX idx ON results(driverId,day)')
out={};examples=[]
with zipfile.ZipFile(P/'sources/l128/driver-dnf.zip') as task:
 for split,n in [('train',11411),('val',566),('test',702)]:
  t=pq.read_table(io.BytesIO(task.read('driver-dnf/'+split+'.parquet')),use_threads=False).to_pandas();assert len(t)==n
  t['day']=t.date.astype('int64')//1000000000
  assert not t.duplicated(['driverId','date']).any()
  for row in t.itertuples():
   value=con.execute('SELECT MAX(CASE WHEN statusId != 1 THEN 1 ELSE 0 END) FROM results WHERE driverId=? AND day>? AND day<=?',(int(row.driverId),int(row.day),int(row.day)+30*86400)).fetchone()[0]
   assert value==row.did_not_finish,(row, value, con.execute('SELECT statusId,day FROM results WHERE driverId=? AND day>? AND day<=?',(int(row.driverId),int(row.day),int(row.day)+30*86400)).fetchall())
  past=sum(con.execute('SELECT EXISTS(SELECT 1 FROM results WHERE driverId=? AND day>? AND day<=?)',(int(row.driverId),int(row.day)-365*86400,int(row.day))).fetchone()[0] for row in t.itertuples())
  out[split]=dict(rows=n,current_positives=int(t.did_not_finish.sum()),historical_positives=int((1-t.did_not_finish).sum()),past_year_eligible=past,no_observed_past_year=n-past)
  if split=='val':
   row=next(t.itertuples());events=con.execute('SELECT statusId,day FROM results WHERE driverId=? AND day>? AND day<=?',(int(row.driverId),int(row.day),int(row.day)+30*86400)).fetchall()
   examples.append(dict(driverId=int(row.driverId),date=str(row.date),current_label=int(row.did_not_finish),historical_label=1-int(row.did_not_finish),future_events=events))
r=dict(status='PASS',method='Independent SQLite future-window aggregation of raw event rows for all 12679 released queries',splits=out,examples=examples,scope='Released statusId != 1 is the target definition, not a corrected motorsport DNF definition; released eligibility query has no upper cutoff',raw_results_rows=len(df))
(P/'_data_l128_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
