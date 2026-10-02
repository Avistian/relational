"""Build exact shared index, then isolate unrelated unfinished lessons if needed.

The receipt distinguishes shared-index failure from isolated-index PASS.
No working files or shared Git index entries are changed by this script.
"""
import hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
def build(root):
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 return subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
with tempfile.TemporaryDirectory(prefix='l184-index-') as tmp:
 root=Path(tmp)/'checkout';root.mkdir();env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'))
 tree=subprocess.check_output(['git','write-tree'],cwd=R,text=True).strip()
 subprocess.run(['git','read-tree',tree],cwd=R,env=env,check=True)
 subprocess.run(['git','checkout-index','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True)
 protected=json.loads((root/'labs/evidence/l184/artifact-manifest.json').read_text())['files']
 for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 result=build(root);shared_status='PASS' if result.returncode==0 else 'BLOCKED_UNRELATED_UNSTAGED_INPUTS';shared_error=result.stderr
 omitted=[]
 if result.returncode:
  manifest=json.loads((root/'lessons/manifest.json').read_text())
  omitted=[x['id'] for x in manifest['lessons'] if x['id']!=184 and (not (root/'lessons'/(x['slug']+'.html')).is_file() or (x.get('labPath') and not (root/x['labPath']).is_file()))]
  # Only omissions of explicitly unpublished-in-index neighbors are permitted.
  assert omitted and all(x in [183,185,186,187,188,189,190] for x in omitted),(omitted,shared_error)
  workflow=(root/'.github/workflows/pages.yml').read_text()
  for n in omitted:workflow=re.sub(r'          # L'+str(n)+r':[^\n]*\n(?:(?!          #)[\s\S])*?(?=          #)', '',workflow)
  manifest['lessons']=[x for x in manifest['lessons'] if x['id'] not in omitted]
  for name,content in [('.github/workflows/pages.yml',workflow),('lessons/manifest.json',json.dumps(manifest,indent=2)+'\n')]:
   blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=content,text=True).strip()
   subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+blob+','+name],cwd=R,env=env,check=True)
  subprocess.run(['git','checkout-index','--force','--all','--prefix='+str(root)+'/'],cwd=R,env=env,check=True)
  result=build(root)
 assert result.returncode==0,result.stderr
 copied=0
 for name,h in protected.items():
  if name.startswith('modal/'):continue
  assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name;copied+=1
 links=0
 for name in ['lessons/0184-gelgt-temporal-attention.html','reference/gelgt-temporal-attention.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',check='Real Pages Build site step from isolated Git index' if omitted else 'Real Pages Build site step from shared Git index',shared_index_status=shared_status,shared_index_error=shared_error,isolated_index_omitted_unready_lessons=omitted,seconds=time.monotonic()-start,lesson=184,authenticated_files=len(protected),authenticated_copied_files=copied,copied_site_links=links,live_deployment='NOT_CHECKED')
(P/'_checkout_l184_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
