"""Pin the exact installed source release and cached complete F1 snapshot."""
import hashlib,json,shutil,urllib.request,zipfile
from importlib.metadata import distribution,version
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l171';E=P/'evidence/l171'
S.mkdir(parents=True,exist_ok=True);E.mkdir(parents=True,exist_ok=True)
package=Path(distribution('relbench').locate_file('relbench'))
assert version('relbench')=='1.1.0','Use the approved source version, not a silent upgrade'
for relative in ['datasets/'+x+'.py' for x in ['amazon','avito','event','f1','hm','stack','trial','__init__']]+['datasets/hashes.json','base/table.py','base/database.py','base/dataset.py']:
    target=S/'relbench'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/relative,target)
# Keep the release's actual LICENSE and metadata, without inferring upstream dataset rights.
dist=distribution('relbench')
for f in dist.files:
    if str(f).endswith(('/LICENSE','/METADATA')):
        target=S/Path(str(f)).name;shutil.copyfile(dist.locate_file(f),target)
cache=Path.home()/'.cache/relbench/rel-f1'
archive=cache/'db.zip';registry=json.loads((S/'relbench/datasets/hashes.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==registry['rel-f1/db.zip']
shutil.copyfile(archive,E/'rel-f1-db.zip')
with zipfile.ZipFile(archive) as z:
    members=[n for n in z.namelist() if n.endswith('.parquet')]
    assert len(members)==9 and all(n.startswith('db/') and '..' not in Path(n).parts for n in members)
    for name in members:
        raw=z.read(name)
        assert raw==(cache/name).read_bytes(),'Cache differs from complete released archive: '+name
        target=E/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
reading=[]
for name,url in [('relbench-paper','https://arxiv.org/html/2407.20060v1'),('relbench-current','https://star-project.stanford.edu/relbench/')]:
    path=S/(name+'.html')
    if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=45).read())
    reading.append(dict(name=name,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
ledger=dict(package='relbench',version=version('relbench'),retrieved='2026-10-02',source_identity='installed release bytes; pinned SHA256; not historical paper byte identity',
    readings=reading,registry_scope='Seven original databases; current expansions deliberately excluded',
    archive=dict(database='rel-f1',url='https://relbench.stanford.edu/download/rel-f1/db.zip',sha256=registry['rel-f1/db.zip'],cache_archive_member_parity='PASS'),
    dataset_rights='REVIEW_REQUIRED: package license is not a substitute for individual source terms')
(S/'source-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
files={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for root in [S,E/'db'] for p in sorted(root.rglob('*')) if p.is_file()}
files['evidence/l171/rel-f1-db.zip']=hashlib.sha256((E/'rel-f1-db.zip').read_bytes()).hexdigest()
(E/'input-manifest.json').write_text(json.dumps(dict(experiment='L171 RelBench Corpus and Holdout Audit',files=files),indent=2)+'\n')
print('Pinned',len(files),'files; complete F1 archive and all nine member tables')
