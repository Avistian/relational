from pathlib import Path
import subprocess,tempfile,json,re,hashlib
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[2];D=R/'reviews/lessons-171-180'
checkout=Path(tempfile.mkdtemp(prefix='review-171-180-index-'))
subprocess.run(['git','checkout-index','--all','--prefix='+str(checkout)+'/'],cwd=R,check=True)
workflow=(checkout/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
p=subprocess.run(['bash','-c',script],cwd=checkout,capture_output=True,text=True);assert p.returncode==0,p.stderr
stage=checkout/'public';files=set();rows=[];cache={}
def soup(p):
 if p not in cache:cache[p]=BeautifulSoup(p.read_text(),'html.parser')
 return cache[p]
for n in [*range(171,179),180]:
 lesson=next((stage/'lessons').glob(f'0{n}-*.html'));pages={lesson,stage/'labs/html'/lesson.name}
 for a in soup(lesson).select('a[href]'):
  u=urlsplit(a['href'])
  if u.path.startswith('../reference/') and u.path.endswith('.html'):pages.add((lesson.parent/u.path).resolve())
 count=0;broken=[]
 for page in pages:
  assert page.exists(),page;files.add(page)
  for el in soup(page).select('[href],[src]'):
   link=el.get('href',el.get('src'));u=urlsplit(link)
   if u.scheme or u.netloc:
    if u.netloc.lower()=='avistian.github.io' and u.path.startswith('/relational/'):
     dest=stage/unquote(u.path[len('/relational/'):])
    else:continue
   else:dest=(page.parent/unquote(u.path)).resolve() if u.path else page
   if not dest.exists():broken.append([str(page.relative_to(stage)),link,'missing file']);continue
   count+=1
   if dest.is_file():files.add(dest)
   if u.fragment and dest.suffix=='.html':
    obj=soup(dest)
    if not any(obj.find(id=f) or obj.find('a',attrs={'name':f}) for f in [u.fragment,unquote(u.fragment)]):broken.append([str(page.relative_to(stage)),link,'missing anchor'])
  if re.search(r'\[\[[A-Z_]+(?::[^\]]+)?\]\]',soup(page).get_text()):broken.append([str(page),'unexpanded marker'])
 rows.append(dict(lesson=n,links=count,pages=len(pages),broken=broken,status='FAIL' if broken else 'PASS'))
for p in ['index.html','notebooks.html','lessons/manifest.json','reviews/lessons-171-180/review.md']:files.add(stage/p)
for n in [*range(171,179),180]:
 files.update((stage/f'labs/figures/l{n}').glob('*'))
 files.add(next((stage/'labs/solutions').glob(f'0{n}-*.ipynb')))
report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',clean_index_build=True,missing_lessons=[179],lessons=rows,total_links=sum(r['links'] for r in rows),stage=str(stage))
(D/'pages.json').write_text(json.dumps(report,indent=2)+'\n')
hashes={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(D/'site-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(report,indent=2));assert report['status']=='PASS';print('Files for live hash verification:',len(hashes))
