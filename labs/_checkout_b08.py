"""Run actual Pages build from isolated Git index; preserve user staging."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
protected=json.loads((P/'evidence/b08/artifact-manifest.json').read_text())['files']
index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
if not index.is_absolute():index=R/index
before=hashlib.sha256(index.read_bytes()).hexdigest()
extras=['.github/workflows/pages.yml','scripts/update-manifest.py','lessons/manifest.json','index.html','notebooks.html','assets/retrieval-pool.js','assets/paper-deck.js','reference/curriculum.html','reference/glossary.html','labs/README.md']
extras += [str(p.relative_to(R)) for p in (P/'evidence/b08').glob('*') if p.is_file()]+[str(p.relative_to(R)) for p in P.glob('_*b08*.json')]
with tempfile.TemporaryDirectory(prefix='b08-publication-') as td:
 tmp=Path(td);new_index=tmp/'index';shutil.copyfile(index,new_index);env=dict(os.environ,GIT_INDEX_FILE=str(new_index),GIT_LFS_SKIP_SMUDGE='1')
 subprocess.run(['git','add','-f','--',*sorted(set(protected)|set(extras))],cwd=R,env=env,check=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True,capture_output=True)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Actual Pages build failed:\n'+result.stderr)
 for name,digest in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==digest,name
 links=0
 for name in ['lessons/b08-structured-objectives.html','reference/b08-structured-objectives.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
assert hashlib.sha256(index.read_bytes()).hexdigest()==before,'User index changed'
r=dict(status='PASS',actual_pages_script='PASS',user_index_unchanged=True,authenticated_copied_files=len(protected),copied_site_links=links,seconds=time.monotonic()-start,live_deployment='NOT_RUN')
(P/'_checkout_b08_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
