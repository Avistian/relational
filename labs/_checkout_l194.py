"""Run actual Pages script from an isolated temporary Git index; preserve user's index."""
import hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
protected=json.loads((P/'evidence/l194/artifact-manifest.json').read_text())['files']
shared=['.gitignore','.github/workflows/pages.yml','lessons/manifest.json','index.html','notebooks.html','assets/retrieval-pool.js','assets/paper-deck.js','reference/glossary.html','reference/curriculum.html']
extras=[str(p.relative_to(R)) for p in P.glob('_*l194*.json')]+['labs/evidence/l194/artifact-manifest.json','labs/evidence/l194/local-budget.json']
original_index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
if not original_index.is_absolute():original_index=R/original_index
original_hash=hashlib.sha256(original_index.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='l194-publication-') as tmp:
 tmp=Path(tmp);index=tmp/'index';shutil.copyfile(original_index,index);env=dict(os.environ,GIT_INDEX_FILE=str(index))
 subprocess.run(['git','add','--',*protected,*shared,*extras],cwd=R,env=env,check=True)
 # Existing workflow also publishes an untracked earlier review package.
 dependency='reviews/lessons-181-190'
 subprocess.run(['git','add','-f','--',dependency],cwd=R,env=env,check=True)
 root=tmp/'checkout';root.mkdir();subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True)
 check=subprocess.run(['python3','labs/_check_pages_checkout.py'],cwd=R,env=env,capture_output=True,text=True)
 if check.returncode:raise AssertionError('Checkout input check failed:\n'+check.stdout+check.stderr)
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Actual Pages build failed:\n'+result.stderr)
 for name,h in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name
 links=0
 for name in ['lessons/0194-open-fm-analysis-report.html','reference/open-fm-analysis-report.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
assert hashlib.sha256(original_index.read_bytes()).hexdigest()==original_hash,'User index changed'
r=dict(status='PASS',actual_pages_script='PASS',checkout_input_check='PASS',index_scope='Temporary copy of existing index plus current L194 deliverables and shared navigation/workflow files',existing_workflow_dependency='reviews/lessons-181-190 included only in temporary index',user_index_unchanged=True,authenticated_copied_files=len(protected),copied_site_links=links,seconds=time.monotonic()-start,live_deployment='NOT_CHECKED')
(P/'_checkout_l194_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
