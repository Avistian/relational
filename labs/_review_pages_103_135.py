"""Crawl reviewed learner pages against a fresh copy of the actual Pages build."""
import argparse,json,re,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
def stage_site():
 stage=Path(tempfile.mkdtemp(prefix='rdl-sequence-pages-'))/'public'
 workflow=(R/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
 lines=[s[10:] for s in block.splitlines() if s.startswith('          ')];lines=[s for s in lines if not s.startswith(('VER=','sed -i'))]
 script='set -eu\n'+'\n'.join(re.sub(r'\bpublic(?=/|\s|$)',str(stage),s) for s in lines)
 subprocess.run(['bash','-c',script],cwd=R,check=True,capture_output=True,text=True)
 return stage

def audit(stage):
 cache={};reports=[]
 def soup(path):
  if path not in cache:cache[path]=BeautifulSoup(path.read_text(),'html.parser')
  return cache[path]
 for n in range(103,136):
  lessons=list((stage/'lessons').glob(f'{n:04d}-*.html'))
  if len(lessons)!=1:reports.append(dict(lesson=n,status='MISSING_LESSON'));continue
  lesson=lessons[0];pages=[lesson,stage/'labs/html'/lesson.name]
  for a in soup(lesson).select('a[href]'):
   u=urlsplit(a['href'])
   if u.path.startswith('../reference/') and u.path.endswith('.html') and 'glossary' not in u.path:pages.append((lesson.parent/u.path).resolve())
  broken=[];count=0
  for page in set(pages):
   if not page.exists():broken.append([str(page.relative_to(stage)),'missing page']);continue
   for el in soup(page).select('[href],[src]'):
    link=el.get('href',el.get('src'));u=urlsplit(link)
    if u.scheme or u.netloc:continue
    dest=(page.parent/unquote(u.path)).resolve() if u.path else page
    if not dest.exists():broken.append([str(page.relative_to(stage)),link,'missing file']);continue
    count+=1
    if u.fragment and dest.suffix=='.html':
     d=soup(dest)
     if not any(d.find(id=f) or d.find('a',attrs={'name':f}) for f in [u.fragment,unquote(u.fragment)]):broken.append([str(page.relative_to(stage)),link,'missing anchor'])
   if re.search(r'\[\[[A-Z_]+(?::[^\]]+)?\]\]',soup(page).get_text()):broken.append([str(page.relative_to(stage)),'unexpanded marker'])
  reports.append(dict(lesson=n,status='PASS' if not broken else 'FAIL',links=count,pages=len(set(pages)),broken=broken))
 result=dict(stage=str(stage),lessons=reports,total_links=sum(r.get('links',0) for r in reports),status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL')
 (R/'reviews/lessons-103-135/pages.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='lessons'},indent=2));print(json.dumps([r for r in reports if r['status']!='PASS'],indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage');a=p.parse_args();audit(Path(a.stage) if a.stage else stage_site())
