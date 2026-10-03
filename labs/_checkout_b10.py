"""Actual copied Pages build through temporary Git index; preserve user staging."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src'))
protected=json.loads((P/'evidence/b10/artifact-manifest.json').read_text())['files']
index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip());index=index if index.is_absolute() else R/index
before=hashlib.sha256(index.read_bytes()).hexdigest()
extras=['.github/workflows/pages.yml','scripts/update-manifest.py','lessons/manifest.json','index.html','notebooks.html','assets/retrieval-pool.js','assets/paper-deck.js','reference/curriculum.html','reference/glossary.html','labs/README.md']
# The existing, uncommitted B08 package is also required by the current Pages workflow.
previous=P/'evidence/b08/artifact-manifest.json'
if previous.exists():extras+=list(json.loads(previous.read_text())['files'])
previous9=P/'evidence/b09/artifact-manifest.json'
if previous9.exists():extras+=list(json.loads(previous9.read_text())['files'])
for b in ['b08','b09','b10']:
 extras += [str(p.relative_to(R)) for p in (P/f'evidence/{b}').rglob('*') if p.is_file()]
 extras += [str(p.relative_to(R)) for p in P.glob(f'_*{b}*.json')]
with tempfile.TemporaryDirectory(prefix='b10-publication-') as td:
 tmp=Path(td);new_index=tmp/'index';shutil.copyfile(index,new_index);env=dict(os.environ,GIT_INDEX_FILE=str(new_index),GIT_LFS_SKIP_SMUDGE='1')
 subprocess.run(['git','add','-f','--',*sorted(set(protected)|set(extras))],cwd=R,env=env,check=True,capture_output=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True,capture_output=True)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Actual Pages build failed:\n'+result.stderr)
 count=0
 for name,digest in protected.items():
  target=root/'public'/name
  # Plans/source markdown are authoring files, not a site deployment promise.
  if not name.startswith(('docs/','lessons/content/','reviews/','learning-records/')) and name not in ['NOTES.md','RESOURCES.md','CURRICULUM.md','thesis-dossier.md']:
   assert target.is_file(),name
   assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,name;count+=1
 links=0
 for name in ['lessons/b10-relational-transformer.html','reference/b10-relational-transformer.html']:
  path=root/'public'/name;p=Links();p.feed(path.read_text())
  for url in p.links:
   u=urlsplit(url)
   if u.scheme or not u.path:continue
   assert (path.parent/unquote(u.path)).resolve().is_file(),url;links+=1
assert hashlib.sha256(index.read_bytes()).hexdigest()==before,'User index changed'
r=dict(status='PASS',actual_pages_script='PASS',user_index_unchanged=True,authenticated_copied_files=count,copied_links=links,seconds=time.monotonic()-start,deployment='NOT_RUN')
(P/'_checkout_b10_results.json').write_text(json.dumps(r,indent=2));print(r)
