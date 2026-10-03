"""Build real Pages from Git index and authenticate all copied L192 evidence."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
with tempfile.TemporaryDirectory(prefix='l192-clean-index-') as tmp:
 root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
 protected=json.loads((root/'labs/evidence/l192/artifact-manifest.json').read_text())['files']
 for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True);assert result.returncode==0,result.stderr
 for name in ['lessons/0192-open-fm-setup-data.html','labs/0192-open-fm-setup-data.ipynb','labs/solutions/0192-open-fm-setup-data.ipynb','labs/html/0192-open-fm-setup-data.html','reference/open-fm-setup-data.html','labs/evidence/l192/report.json','labs/sources/l192/source-ledger.json']:
  assert (root/'public'/name).is_file(),name
 copied=0
 for name,h in protected.items():
  if name.startswith('modal/'):continue # Cloud operator is repository-side, not a site link.
  assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name;copied+=1
 links=0
 for name in ['lessons/0192-open-fm-setup-data.html','reference/open-fm-setup-data.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',check='Complete real Pages build from Git index',seconds=time.monotonic()-start,lesson=192,authenticated_files=len(protected),authenticated_copied_files=copied,copied_site_links=links,live_deployment='NOT_CHECKED')
(P/'_checkout_l192_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
