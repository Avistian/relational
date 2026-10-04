"""Verify the actual Git index and real Pages build; never auto-stage workspace files."""
import hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
P=Path(__file__).resolve().parent;R=P.parent

def main():
 started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='b23-index-pages-') as td:
  root=Path(td)
  subprocess.run(['git','checkout-index','--all','--prefix='+td+'/'],cwd=R,env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'),check=True,capture_output=True)
  subprocess.run([sys.executable,'labs/_test_b05_source_secrets.py'],cwd=root,check=True,capture_output=True)
  workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
  script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
  proc=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True);assert proc.returncode==0,proc.stderr
  files=[]
  for p in root.rglob('*'):
   if not p.is_file() or 'public' in p.relative_to(root).parts:continue
   rel=p.relative_to(root);parts=rel.parts
   selected=('b23' in str(rel) or p.name in ['comparison-evidence.css','comparison-evidence.js','b23-results.js'])
   if not selected or parts[0] in ['reviews','learning-records'] or parts[:2]==('lessons','content'):continue
   target=root/'public'/rel;assert target.is_file(),str(rel)
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest(),str(rel)
   files.append(str(rel))
  class Links(HTMLParser):
   def __init__(self):super().__init__();self.urls=[]
   def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
  links=0
  for slug in ['b23-declared-comparison']:
   for folder in ['lessons','reference']:
    file=root/'public'/folder/(slug+'.html');assert file.is_file();parser=Links();parser.feed(file.read_text())
    for url in parser.urls:
     u=urlsplit(url)
     if u.scheme:continue
     target=(file.parent/unquote(u.path)).resolve() if u.path else file
     assert target.is_file(),(str(file),url);links+=1
  manifest=json.loads((root/'public/lessons/manifest.json').read_text());assert manifest['version']==19
  assert all(any(x['id']==i for x in manifest['lessons']) for i in ['B23'])
  site_bytes=sum(p.stat().st_size for p in (root/'public').rglob('*') if p.is_file())
  out=dict(status='PASS',actual_git_index=True,actual_workflow_build=True,archived_source_secret_check=True,byte_identical_b23_files=len(files),files=sorted(files),copied_site_local_links=links,site_bytes=site_bytes,seconds=time.monotonic()-started)
  if '--check-only' not in sys.argv:(P/'_pages_b23_results.json').write_text(json.dumps(out,indent=2)+'\n')
  print({k:v for k,v in out.items() if k!='files'})
if __name__=='__main__':main()
