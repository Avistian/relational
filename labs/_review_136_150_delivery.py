"""Build only indexed files, then audit every reviewed learner page over HTTP."""
import functools,json,os,re,subprocess,tempfile,threading
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from _gallery_delivery import reveal_gallery_link
R=Path(__file__).resolve().parents[1];D=R/'reviews/lessons-136-150'
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
checkout=Path(tempfile.mkdtemp(prefix='review-136-149-index-'))
subprocess.run(['git','checkout-index','--all','--prefix='+str(checkout)+'/'],cwd=R,check=True)
workflow=(checkout/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
built=subprocess.run(['bash','-c',script],cwd=checkout,capture_output=True,text=True)
assert built.returncode==0,built.stderr
stage=checkout/'public';cache={};reports=[]
def soup(p):
 if p not in cache:cache[p]=BeautifulSoup(p.read_text(),'html.parser')
 return cache[p]
for n in range(136,150):
 lesson=next((stage/'lessons').glob(f'{n:04}-*.html'));pages={lesson,stage/'labs/html'/lesson.name}
 for a in soup(lesson).select('a[href]'):
  u=urlsplit(a['href'])
  if u.path.startswith('../reference/') and u.path.endswith('.html'):pages.add((lesson.parent/u.path).resolve())
 broken=[];count=0
 for page in pages:
  if not page.exists():broken.append([str(page),'missing page']);continue
  for el in soup(page).select('[href],[src]'):
   link=el.get('href',el.get('src'));u=urlsplit(link)
   if u.scheme or u.netloc:continue
   dest=(page.parent/unquote(u.path)).resolve() if u.path else page
   if not dest.exists():broken.append([str(page.relative_to(stage)),link,'missing file']);continue
   count+=1
   if u.fragment and dest.suffix=='.html':
    obj=soup(dest)
    if not any(obj.find(id=f) or obj.find('a',attrs={'name':f}) for f in [u.fragment,unquote(u.fragment)]):broken.append([str(page.relative_to(stage)),link,'missing anchor'])
  if re.search(r'\[\[[A-Z_]+(?::[^\]]+)?\]\]',soup(page).get_text()):broken.append([str(page),'unexpanded marker'])
 reports.append(dict(lesson=n,links=count,pages=len(pages),broken=broken,status='FAIL' if broken else 'PASS'))
report=dict(status='PASS' if all(x['status']=='PASS' for x in reports) else 'FAIL',clean_index_build=True,lessons=reports,total_links=sum(x['links'] for x in reports),stage=str(stage))
(D/'pages.json').write_text(json.dumps(report,indent=2)+'\n');assert report['status']=='PASS',reports
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
rows=[];errors=[];http_errors=[];screens=Path('/tmp/review-136-149-screens');screens.mkdir(exist_ok=True)
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
  page.on('response',lambda r:http_errors.append([r.status,r.url]) if r.url.startswith(base) and r.status>=400 else None)
  page.goto(base+'/index.html')
  for n in range(136,150):
   path=next((stage/'lessons').glob(f'{n:04d}-*.html'));reveal_gallery_link(page,f'a[href="lessons/{path.name}"]')
  for n in range(136,150):
   path=next((stage/'lessons').glob(f'{n:04d}-*.html'));url=base+'/lessons/'+path.name
   for width in [1200,375]:
    page.set_viewport_size(dict(width=width,height=900));page.goto(url);page.wait_for_timeout(80)
    assert page.locator('h1').is_visible();assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(n,width,'overflow')
    imgs=page.locator('figure img');assert imgs.evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)'),(n,width,'images')
    reminder=page.locator('.sequence-context summary');reminder.focus();page.keyboard.press('Enter');assert page.locator('.sequence-context details').get_attribute('open') is not None
    page.evaluate('scrollTo(0,0)');page.screenshot(path=str(screens/f'{n}-{width}-top.png'))
    for i in range(page.locator('figure').count()):page.locator('figure').nth(i).screenshot(path=str(screens/f'{n}-{width}-figure-{i}.png'))
    rows.append(dict(lesson=n,width=width,status='PASS',figures=imgs.count(),no_page_overflow=True))
   page.emulate_media(media='print');assert page.locator('h1').is_visible();assert imgs.evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)');page.emulate_media(media='screen')
   context=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900));p=context.new_page();p.goto(url)
   assert p.locator('h1').is_visible();assert not p.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(n,'noJS overflow');context.close()
   print(n,'desktop/mobile, figures, keyboard, print, no-JS PASS',flush=True)
  # The new route diagram is also inspected at its real SVG dimensions.
  page.goto(base+'/labs/figures/l141/architecture.svg')
  escaped=page.locator('text').evaluate_all('xs=>xs.filter(e=>{let b=e.getBBox();return b.x<0||b.y<0||b.x+b.width>1100||b.y+b.height>904}).map(e=>e.textContent)')
  assert not escaped,escaped
  browser.close()
finally:server.shutdown();server.server_close();thread.join()
report=dict(status='PASS' if not errors and not http_errors else 'FAIL',views=rows,print_lessons=14,nojs_lessons=14,manifest_links=14,javascript_errors=errors,http_errors=http_errors,stage=str(stage),screenshots=str(screens),new_svg_labels_in_bounds=True)
(D/'browser.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='views'});assert report['status']=='PASS'
