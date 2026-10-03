"""Build actual Pages inputs via a temporary index; stage only declared packages."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src'))
index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip());index=index if index.is_absolute() else R/index
before=hashlib.sha256(index.read_bytes()).hexdigest()
extras=['.github/workflows/pages.yml','.gitignore','CURRICULUM.md','NOTES.md','RESOURCES.md','thesis-dossier.md','requirements-labs.txt','scripts/update-manifest.py','lessons/manifest.json','index.html','notebooks.html','assets/retrieval-pool.js','assets/paper-deck.js','reference/curriculum.html','reference/glossary.html','labs/README.md','plan/year-5-6-bridge.md']
protected={}
for b in range(8,16):
 name=f'b{b:02d}';m=json.loads((P/f'evidence/{name}/artifact-manifest.json').read_text())['files']
 for f,digest in m.items():
  assert hashlib.sha256((R/f).read_bytes()).hexdigest()==digest,'Changed sealed artifact: '+f
 protected.update(m)
 extras += [str(p.relative_to(R)) for p in (P/f'evidence/{name}').rglob('*') if p.is_file()]
 extras += [str(p.relative_to(R)) for p in P.glob(f'_*{name}*.json')]
 # Keep existing integration/build helpers with their authored packages.
 extras += [str(p.relative_to(R)) for p in P.glob(f'_*{name}*.py')]
extras += [str(p.relative_to(R)) for p in (R/'learning-records').glob('01[67]*-*-prepared.md')]
files=sorted(set(extras)|set(protected))
with tempfile.TemporaryDirectory(prefix='b15-publication-') as td:
 tmp=Path(td);new_index=tmp/'index';shutil.copyfile(index,new_index);env=dict(os.environ,GIT_INDEX_FILE=str(new_index),GIT_LFS_SKIP_SMUDGE='1')
 subprocess.run(['git','add','-f','--',*files],cwd=R,env=env,check=True,capture_output=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True,capture_output=True)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Actual Pages build failed:\n'+result.stderr)
 count=0
 for name,digest in protected.items():
  target=root/'public'/name
  if name.startswith(('labs/','lessons/','reference/','assets/','modal/')) and not name.startswith('lessons/content/'):
   assert target.is_file(),'Missing publication dependency: '+name
   assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,name;count+=1
 links=0
 for path in list((root/'public/lessons').glob('b*.html'))+list((root/'public/reference').glob('b*.html')):
  p=Links();p.feed(path.read_text())
  for url in p.links:
   u=urlsplit(url)
   if u.scheme or not u.path:continue
   assert (path.parent/unquote(u.path)).resolve().is_file(),str(path)+': '+url;links+=1
 size=sum(p.stat().st_size for p in (root/'public').rglob('*') if p.is_file())
assert hashlib.sha256(index.read_bytes()).hexdigest()==before,'User index changed'
# Explicit reviewed closure is also the staging input for the authorized push.
(P/'evidence/b15/publication-files.json').write_text(json.dumps(files,indent=2)+'\n')
r=dict(status='PASS',actual_pages_script='PASS',user_index_unchanged=True,authenticated_copied_files=count,bridge_links=links,public_bytes=size,seconds=time.monotonic()-start,deployment='PENDING')
(P/'_checkout_b15_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
