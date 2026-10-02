"""Build the real Pages workflow from the index; verify L183 copied bytes and links."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic();S='0183-graph-transformer-pretraining'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
with tempfile.TemporaryDirectory(prefix='l183-clean-index-') as tmp:
 root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
 protected=json.loads((root/'labs/evidence/l183/artifact-manifest.json').read_text())['files']
 for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
 result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True);assert result.returncode==0,result.stderr
 for name,h in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name
 entry=next(x for x in json.loads((root/'public/lessons/manifest.json').read_text())['lessons'] if x['id']==183)
 assert entry['published'] and entry['labPath']=='labs/'+S+'.ipynb' and (root/'public'/entry['labPath']).is_file()
 links=0
 for name in ['lessons/'+S+'.html','reference/graph-transformer-pretraining.html']:
  path=root/'public'/name;parser=Links();parser.feed(path.read_text())
  for url in parser.links:
   part=urlsplit(url)
   if part.scheme or not part.path:continue
   assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
r=dict(status='PASS',check='Complete real Pages build from Git index',seconds=time.monotonic()-start,lesson=183,authenticated_files=len(protected),authenticated_copied_files=len(protected),copied_site_links=links,live_deployment='NOT_CHECKED')
(P/'_checkout_l183_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
