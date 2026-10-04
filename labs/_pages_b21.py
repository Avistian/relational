"""Run the real Pages build from a temporary Git index; preserve the real index."""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
P=Path(__file__).resolve().parent;R=P.parent;S='b21-structural-robustness'
def main():
 started=time.monotonic();index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip());index=index if index.is_absolute() else R/index;before=hashlib.sha256(index.read_bytes()).hexdigest()
 paths=[]
 for root in [P/'evidence/b21',P/'sources/b21',P/'figures/b21']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 for pattern in ['_*b21*.py','_*b21*.json','b21-*.md','b21-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/structural_b21.py',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference'/f'{S}.html',R/'assets/structural-robustness.css',R/'assets/structural-robustness.js',R/'docs/plans/2026-10-04-b21-design.md'])
 with tempfile.TemporaryDirectory(prefix='b21-pages-') as td:
  t=Path(td);env=dict(os.environ,GIT_INDEX_FILE=str(t/'index'),GIT_LFS_SKIP_SMUDGE='1')
  def git(*args):return subprocess.run(['git',*args],cwd=R,env=env,check=True,capture_output=True,text=True)
  git('read-tree','HEAD');git('add','-u');git('add','-f','--',*[str(p.relative_to(R)) for p in sorted(set(paths))])
  prerequisite=[]
  for folder in ['labs/evidence/b20','labs/sources/b20','labs/figures/b20']:
   prerequisite.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
  for pattern in ['_*b20*.py','_*b20*.json','b20-*.md','b20-*.ipynb']:prerequisite.extend(P.glob(pattern))
  prerequisite.extend([P/'relkit/curriculum_b20.py',P/'solutions/b20-curriculum-order.ipynb',P/'html/b20-curriculum-order.html',R/'lessons/b20-curriculum-order.html',R/'lessons/content/b20-curriculum-order.md',R/'reference/b20-curriculum-order.html',R/'assets/curriculum-order.js',R/'assets/curriculum-order.css'])
  git('add','-f','--',*[str(p.relative_to(R)) for p in prerequisite])
  root=t/'checkout';root.mkdir();git('checkout-index','--all','--prefix='+str(root)+'/')
  workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
  script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
  proc=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
  assert proc.returncode==0,proc.stderr
  checked=[]
  for p in sorted(set(paths)):
   rel=p.relative_to(R)
   if rel.parts[:2]==('lessons','content') or p.name in ['local-budget.json','_pages_b21_results.json']:continue
   target=root/'public'/rel;assert target.is_file(),str(rel)
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest(),str(rel)
   checked.append(str(rel))
  class Links(HTMLParser):
   def __init__(self):super().__init__();self.urls=[]
   def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
  links=0
  for rel in [f'lessons/{S}.html',f'reference/{S}.html']:
   file=root/'public'/rel;parser=Links();parser.feed(file.read_text())
   for url in parser.urls:
    u=urlsplit(url)
    if u.scheme:continue
    target=(file.parent/unquote(u.path)).resolve() if u.path else file
    assert target.is_file(),url;links+=1
  manifest=json.loads((root/'public/lessons/manifest.json').read_text());assert any(x['id']=='B21' and x['labPath']==f'labs/{S}.ipynb' for x in manifest['lessons'])
  assert hashlib.sha256(index.read_bytes()).hexdigest()==before,'Real index changed'
  out=dict(status='PASS',actual_workflow_build=True,temporary_index=True,user_index_preserved=True,byte_identical_b21_files=len(checked),files=checked,copied_site_local_links=links,seconds=time.monotonic()-started,publication='NOT_REQUESTED')
  (P/'_pages_b21_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='files'})
if __name__=='__main__':main()
