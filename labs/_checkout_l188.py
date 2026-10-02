"""Build real Pages from staged Git contents and authenticate every fixed L188 artifact.
If unrelated L183 staging is unfinished, an isolated copy-only build is reported
separately. The user's Git index and working workflow are never altered here.
"""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
R=Path(__file__).resolve().parents[1];P=R/'labs'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src','srcset'])
def check():
 start=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='l188-clean-index-') as tmp:
  root=Path(tmp);subprocess.run(['git','checkout-index','--all','--prefix='+tmp+'/'],cwd=R,check=True)
  protected=json.loads((root/'labs/evidence/l188/artifact-manifest.json').read_text())['files']
  for name,h in protected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
  workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
  def run(text):
   script='set -eu\n'+'\n'.join(line[10:] for line in text.splitlines() if line.startswith('          '))
   return subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
  result=run(block);shared='PASS' if result.returncode==0 else 'FAIL';error=result.stderr if result.returncode else None;isolated=False
  if result.returncode:
   marker='          # L183: portable saved-evidence replay and retained blocked paper protocols.'
   if marker not in block or "cannot stat 'labs/l183-*.md'" not in result.stderr:raise AssertionError(result.stderr)
   begin=block.index(marker);end=block.index('\n          # ',begin+len(marker));isolated=True
   result=run(block[:begin]+block[end:]);assert result.returncode==0,result.stderr
  copied=0
  for name,h in protected.items():
   assert hashlib.sha256((root/'public'/name).read_bytes()).hexdigest()==h,name;copied+=1
  links=0
  for name in ['lessons/0188-systematic-literature-tracking.html','reference/systematic-literature-tracking.html']:
   path=root/'public'/name;parser=Links();parser.feed(path.read_text())
   for url in parser.links:
    part=urlsplit(url)
    if part.scheme or not part.path:continue
    assert (path.parent/unquote(part.path)).resolve().is_file(),url;links+=1
 r=dict(status='PASS_ISOLATED_ONLY' if isolated else 'PASS',shared_git_index_build=shared,shared_error=error,isolated_change='Omitted unfinished L183 copy block only in temporary build script' if isolated else None,seconds=time.monotonic()-start,lesson=188,authenticated_files=len(protected),authenticated_copied_files=copied,copied_site_links=links,live_deployment='NOT_CHECKED')
 (P/'_checkout_l188_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':check()
