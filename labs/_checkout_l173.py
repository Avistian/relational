"""Build real Pages from the Git index and verify all L173 evidence and links."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
with tempfile.TemporaryDirectory(prefix='l173-clean-index-') as tmp:
    root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
    manifest=json.loads((root/'labs/evidence/l173/manifest.json').read_text());seal=json.loads((root/'labs/evidence/l173/artifact-manifest.json').read_text())
    protected={**{'labs/'+n:h for n,h in manifest['inputs'].items()},**{'labs/'+n:h for n,h in manifest['source_files'].items()},**{'labs/evidence/l173/'+n:h for n,h in seal['files'].items()}}
    for name,digest in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
    script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
    result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True);assert result.returncode==0,result.stderr
    for name in ['lessons/0173-multi-task-pretraining.html','labs/0173-multi-task-pretraining.ipynb','labs/solutions/0173-multi-task-pretraining.ipynb','labs/html/0173-multi-task-pretraining.html','reference/multi-task-pretraining.html','labs/evidence/l173/report.json','labs/sources/l173/source-ledger.json']:
        assert (root/'public'/name).is_file(),name
    for name,digest in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==digest,name
    links=0
    for name in ['lessons/0173-multi-task-pretraining.html','reference/multi-task-pretraining.html']:
        path=root/'public'/name;parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            dest=(path.parent/unquote(part.path)).resolve();assert dest.is_file(),str(dest);links+=1
r=dict(status='PASS',check='Complete real Pages build from Git index',seconds=time.monotonic()-start,lesson=173,authenticated_files=len(protected),copied_site_links=links,live_deployment='NOT_CHECKED')
(P/'_checkout_l173_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
