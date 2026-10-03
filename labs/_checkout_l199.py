"""Actual Pages build from temporary Git index; never stage unrelated user work."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
protected=json.loads((P/'evidence/l199/artifact-manifest.json').read_text())['files']
shared=['.gitignore','.github/workflows/pages.yml','lessons/manifest.json','index.html','notebooks.html','reference/curriculum.html','labs/README.md']
extras=[str(p.relative_to(R)) for p in P.glob('_*l199*.json')]+['labs/evidence/l199/artifact-manifest.json','labs/evidence/l199/local-budget.json']
original=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
if not original.is_absolute():original=R/original
before=hashlib.sha256(original.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='l199-publication-') as td:
 tmp=Path(td);index=tmp/'index';shutil.copyfile(original,index);env=dict(os.environ,GIT_INDEX_FILE=str(index))
 subprocess.run(['git','add','-f','--',*protected,*shared,*extras],cwd=R,env=env,check=True)
 # Existing workflow includes active predecessor packages absent from real index.
 dependencies=set()
 for n in range(194,199):
  seal=P/f'evidence/l{n}/artifact-manifest.json'
  if seal.exists():dependencies.update(json.loads(seal.read_text())['files'])
  else:
   for pattern in [f'labs/*l{n}*',f'labs/relkit/*l{n}*',f'labs/019{n%10}-*',f'labs/solutions/019{n%10}-*',f'labs/html/019{n%10}-*',f'lessons/019{n%10}-*',f'lessons/content/019{n%10}-*']:
    dependencies.update(str(p.relative_to(R)) for p in R.glob(pattern) if p.is_file())
   for dirname in [f'labs/evidence/l{n}',f'labs/figures/l{n}']:
    dependencies.update(str(p.relative_to(R)) for p in (R/dirname).rglob('*') if p.is_file() and '__pycache__' not in str(p))
  dependencies.update(str(p.relative_to(R)) for p in P.glob(f'_*l{n}*.json'))
  for name in ['artifact-manifest.json','local-budget.json']:
   if (P/f'evidence/l{n}'/name).exists():dependencies.add(f'labs/evidence/l{n}/'+name)
 # Include current predecessor assets/reference files if that lesson is still preparing its seal.
 for name in ['assets/proposal-decisions.js','assets/proposal-decisions.css','reference/three-research-directions.html']:
  if (R/name).is_file():dependencies.add(name)
 dependencies.add('reviews/lessons-181-190')
 subprocess.run(['git','add','-f','--',*sorted(dependencies)],cwd=R,env=env,check=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 proc=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if proc.returncode:raise AssertionError('Actual Pages build failed:\n'+proc.stderr)
 for name,digest in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==digest,name
 links=0
 for name in ['lessons/0199-select-primary-direction.html','reference/select-primary-direction.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
assert hashlib.sha256(original.read_bytes()).hexdigest()==before,'User index changed during check'
r=dict(status='PASS',actual_pages_script='PASS',index_scope='Temporary copy of user index plus L199, current shared navigation and active L194-L198 dependencies',user_index_unchanged=True,authenticated_copied_files=len(protected),copied_site_links=links,seconds=time.monotonic()-start,live_deployment='NOT_CHECKED')
(P/'_checkout_l199_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
