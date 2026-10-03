"""Actual workflow build from a temporary Git index; preserve the user's index."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
protected=json.loads((P/'evidence/b01/artifact-manifest.json').read_text())['files']
original=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
if not original.is_absolute():original=R/original
before=hashlib.sha256(original.read_bytes()).hexdigest()
extras=['.github/workflows/pages.yml','labs/evidence/b01/artifact-manifest.json','labs/evidence/b01/local-budget.json']+[str(p.relative_to(R)) for p in P.glob('_*b01*.json')]
with tempfile.TemporaryDirectory(prefix='b01-publication-') as td:
 tmp=Path(td);index=tmp/'index';shutil.copyfile(original,index);env=dict(os.environ,GIT_INDEX_FILE=str(index))
 subprocess.run(['git','add','-f','--',*protected,*extras],cwd=R,env=env,check=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True,capture_output=True)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Actual Pages build failed:\n'+result.stderr)
 for name,digest in protected.items():
  assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==digest,name
 links=0
 for name in ['lessons/b01-architecture-coverage-honest-comparison.html','reference/b01-comparison-contract.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
assert hashlib.sha256(original.read_bytes()).hexdigest()==before,'User index changed'
r=dict(status='PASS',actual_pages_script='PASS',isolated_index='Original index plus B01 and modified shared publication inputs',user_index_unchanged=True,authenticated_copied_files=len(protected),copied_site_links=links,seconds=time.monotonic()-start,live_deployment='NOT_CHECKED')
(P/'_checkout_b01_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
