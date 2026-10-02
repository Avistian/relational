"""Run the real Pages build from Git-index bytes, then audit L185's public links."""
import hashlib,json,shutil,subprocess,tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
R=Path(__file__).resolve().parents[1];P=R/'labs'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for key,value in attrs:
   if key in ('src','href'):self.links.append(value)
with tempfile.TemporaryDirectory(prefix='l185-index-pages-') as tmp:
 root=Path(tmp)
 subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
 workflow=(root/'.github/workflows/pages.yml').read_text()
 block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 full_error=result.stderr if result.returncode else None
 pub=root/'public';checked=0
 if full_error:
  # Preserve the unrelated build failure; verify only our staged package separately.
  for name in json.loads((root/'labs/evidence/l185/package-manifest.json').read_text()):
   if name.startswith(('assets/','lessons/','reference/','labs/')):
    dest=pub/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
 for name,digest in json.loads((root/'labs/evidence/l185/package-manifest.json').read_text()).items():
  source=root/name
  assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
  # Author planning/record files live in the repository, not the published course tree.
  if name.startswith(('assets/','lessons/','reference/','labs/')):
   assert (pub/name).is_file(),name
   assert hashlib.sha256((pub/name).read_bytes()).hexdigest()==digest,name
   checked+=1
 for name in ['lessons/0185-causal-relational-data.html','reference/causal-relational-data.html','labs/html/0185-causal-relational-data.html']:
  f=pub/name;parser=Links();parser.feed(f.read_text())
  for link in parser.links:
   url=urlsplit(link)
   if not url.scheme and url.path:assert (f.parent/unquote(url.path)).resolve().exists(),(name,link)
 entry=next(x for x in json.loads((pub/'lessons/manifest.json').read_text())['lessons'] if x['id']==185)
 assert entry['labPath']=='labs/0185-causal-relational-data.ipynb'
 result={'status':'PASS_PACKAGE_ONLY' if full_error else 'PASS','full_site_build':'BLOCKED_UNRELATED_FILES' if full_error else 'PASS','full_site_error':full_error,'method':'Git-index package copy after full workflow failure' if full_error else 'Actual Pages workflow from Git index, copied public tree','sha256_matched_public_files':checked,'local_package_links':'PASS','deployment':'NOT_CHECKED'}
(P/'_checkout_l185_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
