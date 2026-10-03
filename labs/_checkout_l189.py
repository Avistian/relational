"""Run the actual Pages build from staged bytes and verify every sealed artifact."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src','srcset'])
with tempfile.TemporaryDirectory(prefix='l189-clean-index-') as tmp:
 root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
 protected=json.loads((root/'labs/evidence/l189/artifact-manifest.json').read_text())['files']
 for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 for name,h in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name
 count=0
 for path in [root/'public/lessons/0189-identify-open-problems.html',root/'public/reference/identify-open-problems.html']:
  parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;count+=1
result=dict(status='PASS',build='ACTUAL_WORKFLOW_FROM_GIT_INDEX',sealed_files=len(protected),copied_hashes=len(protected),copied_local_links=count,seconds=time.monotonic()-start,cloud_usd=0,deployment='NOT_CHECKED')
(P/'_checkout_l189_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
