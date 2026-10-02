"""Run the actual Pages script from Git index, then authenticate copied L190 bytes."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
with tempfile.TemporaryDirectory(prefix='l190-clean-index-') as tmp:
 root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
 protected=json.loads((root/'labs/evidence/l190/artifact-manifest.json').read_text())['files']
 for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 if result.returncode:raise AssertionError('Real Git-index Pages build failed:\n'+result.stderr)
 for name,h in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name
 links=0
 for name in ['lessons/0190-research-gap-checkpoint.html','reference/research-gap-checkpoint.html','reference/research-gap-document.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',real_git_index_build='PASS',authenticated_files=len(protected),authenticated_copied_files=len(protected),copied_site_links=links,seconds=time.monotonic()-start,live_deployment='NOT_CHECKED')
(P/'_checkout_l190_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
