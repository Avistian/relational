"""Actual Pages build from Git index; verify every sealed lesson-owned copied file."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs';start=time.monotonic()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
with tempfile.TemporaryDirectory(prefix='l187-clean-index-') as tmp:
    root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
    protected=json.loads((root/'labs/evidence/l187/artifact-manifest.json').read_text())['files']
    for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
    script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
    result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
    if result.returncode:
        failure={'status':'FAIL_PUBLICATION_BUILD','lesson':187,'lesson_artifact_hashes':'PASS',
            'workflow_sha256':hashlib.sha256(workflow.encode()).hexdigest(),'error':result.stderr,
            'seconds':time.monotonic()-start,'live_deployment':'NOT_CHECKED'}
        (P/'_checkout_l187_results.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise AssertionError(result.stderr)
    for name,h in protected.items():assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name
    links=0
    for name in ['lessons/0187-ethics-privacy-reg.html','reference/ethics-privacy-reg.html']:
        path=root/'public'/name;parser=Links();parser.feed(path.read_text())
        for url in parser.links:
            part=urlsplit(url)
            if part.scheme or not part.path:continue
            assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
result={'status':'PASS','check':'Complete real Pages build from Git index','seconds':time.monotonic()-start,
    'lesson':187,'authenticated_files':len(protected),'authenticated_copied_files':len(protected),'copied_site_links':links,
    'workflow_sha256':hashlib.sha256(workflow.encode()).hexdigest(),
    'live_deployment':'NOT_CHECKED'}
(P/'_checkout_l187_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
