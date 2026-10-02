"""Pin complete input population and explicit, reviewed course column policies."""
import hashlib,json,zipfile
from pathlib import Path
from _audit_l171 import load_f1
P=Path(__file__).resolve().parent;E=P/'evidence/l172';E.mkdir(exist_ok=True,parents=True)
prior=json.loads((P/'evidence/l171/input-manifest.json').read_text())
files={k:v for k,v in prior['files'].items() if k.startswith('evidence/l171/') or k in ['sources/l171/relbench/datasets/f1.py','sources/l171/relbench/datasets/hashes.json']}
for k,v in files.items():assert hashlib.sha256((P/k).read_bytes()).hexdigest()==v
with zipfile.ZipFile(P/'evidence/l171/rel-f1-db.zip') as z:
 for n in z.namelist():
  if n.endswith('.parquet'):assert z.read(n)==(P/'evidence/l171'/n).read_bytes()
policies={
 'circuits':dict(circuitId='key',circuitRef='text',name='text',location='text',country='category',lat='number',lng='number',alt='number'),
 'constructor_results':dict(constructorResultsId='key',raceId='key',constructorId='key',points='number',date='timestamp'),
 'constructor_standings':dict(constructorStandingsId='key',raceId='key',constructorId='key',points='number',position='number',wins='number',date='timestamp'),
 'constructors':dict(constructorId='key',constructorRef='text',name='text',nationality='category'),
 'drivers':dict(driverId='key',driverRef='text',code='category',forename='text',surname='text',dob='timestamp',nationality='category'),
 'qualifying':dict(qualifyId='key',raceId='key',driverId='key',constructorId='key',number='category',position='number',date='timestamp'),
 'races':dict(raceId='key',year='number',round='number',circuitId='key',name='text',date='timestamp',time='text'),
 'results':dict(resultId='key',raceId='key',driverId='key',constructorId='key',number='category',grid='number',position='number',positionOrder='number',points='number',laps='number',milliseconds='number',fastestLap='number',rank='number',statusId='category',date='timestamp'),
 'standings':dict(driverStandingsId='key',raceId='key',driverId='key',points='number',position='number',wins='number',date='timestamp')}
schema={}
for n,t in load_f1(P).items():
 assert set(t['df'])==set(policies[n])
 columns={}
 for c,kind in sorted(policies[n].items()):
  role='primary_key' if c==t['pkey_col'] else 'foreign_key' if c in t['fkey_col_to_pkey_table'] else 'event_time' if c==t['time_col'] else 'feature'
  columns[c]=dict(kind=kind,role=role,storage_dtype=str(t['df'][c].dtype),schema_name=c+' of '+n)
  if role=='foreign_key':columns[c]['target_table']=t['fkey_col_to_pkey_table'][c]
 schema[n]=dict(time_col=t['time_col'],fit_cutoff='2005-01-01',untimed_fit='NO_ROWS_ADMITTED',columns=columns)
(E/'schema.json').write_text(json.dumps(schema,indent=2,sort_keys=True)+'\n')
files['evidence/l172/schema.json']=hashlib.sha256((E/'schema.json').read_bytes()).hexdigest()
for p in (P/'sources/l172').glob('*'):files[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
(E/'input-manifest.json').write_text(json.dumps(dict(experiment='L172 Full F1 Schema-Tokenization Audit',files=files),indent=2)+'\n')
print('Explicit policies:',sum(len(t['columns']) for t in schema.values()),'columns in',len(schema),'tables')
