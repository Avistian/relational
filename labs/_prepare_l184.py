"""Assemble immutable local inputs; cached archives must match known hashes."""
from pathlib import Path
import hashlib,json,shutil,inspect,zipfile
from relbench.tasks.f1 import DriverPositionTask
P=Path(__file__).resolve().parent;E=P/'evidence/l184';packet=E/'packet';packet.mkdir(exist_ok=True)
cache=Path('/home/avist/.cache/relbench/rel-f1')
archives={'db.zip':'ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482','tasks/driver-position.zip':'775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e'}
for f,h in archives.items():assert hashlib.sha256((cache/f).read_bytes()).hexdigest()==h
for archive,prefix,directory in [('db.zip','db/','db'),('tasks/driver-position.zip','driver-position/','tasks/driver-position')]:
 with zipfile.ZipFile(cache/archive) as z:
  for name in z.namelist():
   if name.endswith('.parquet'):assert z.read(name)==(cache/directory/Path(name).name).read_bytes(),name
for source,target in [(P/'sources/l184/upstream',packet/'upstream'),(cache/'db',packet/'db'),(cache/'tasks/driver-position',packet/'task')]:
 target.mkdir(exist_ok=True)
 for f in source.glob('*'):
  if f.is_file():shutil.copy2(f,target/f.name)
(packet/'task-definition.py').write_text(inspect.getsource(DriverPositionTask))
m={'data_provenance':'Authenticated locally cached RelBench F1 archives, matches earlier source audit; GelGT historical archive identity unestablished','archives':archives,'files':{str(f.relative_to(packet)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(packet.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}}
(E/'input-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(len(m['files']),'authenticated files')
