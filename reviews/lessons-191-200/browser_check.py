from pathlib import Path
import sys,os,json,functools,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[2];D=R/'reviews/lessons-191-200';sys.path.insert(0,str(R/'labs'));from _gallery_delivery import reveal_gallery_link
libs=Path('/tmp/relational-browser-libs/usr/lib/aarch64-linux-gnu')
if libs.exists():os.environ['LD_LIBRARY_PATH']=str(libs)+':'+os.environ.get('LD_LIBRARY_PATH','')
stage=Path(json.loads((D/'pages.json').read_text())['stage']);live='--live' in sys.argv
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=None
if live:base='https://avistian.github.io/relational'
else:
 server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(stage)));threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
rows=[];errors=[];http=[];failed=[];screens=Path('/tmp/review-191-200-'+('live' if live else 'local'));screens.mkdir(exist_ok=True)
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage']);page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 page.on('response',lambda r:http.append([r.status,r.url]) if r.status>=400 and r.url.startswith(base) else None)
 page.on('requestfailed',lambda r:failed.append(r.url) if r.url.startswith(base) else None)
 for gallery,prefix in [('index.html','lessons/'),('notebooks.html','labs/html/')]:
  page.goto(base+'/'+gallery,wait_until='networkidle')
  for n in list(range(191,201)):
   lesson=next((stage/'lessons').glob(f'0{n}-*.html'));reveal_gallery_link(page,f'a[href="{prefix}{lesson.name}"]')
 for n in list(range(191,201)):
  lesson=next((stage/'lessons').glob(f'0{n}-*.html'))
  for width in [1200,375]:
   page.set_viewport_size(dict(width=width,height=900));r=page.goto(base+'/lessons/'+lesson.name,wait_until='networkidle');assert r.status==200
   assert page.locator('h1').first.is_visible();assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),(n,width,'overflow')
   assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)'),(n,'images')
   assert page.locator('h2').count()>=5
   if width==1200:assert page.locator('figure').evaluate_all('(xs)=>xs.every(x=>x.scrollWidth<=x.clientWidth+1)'),(n,'desktop figure clipped')
   page.screenshot(path=str(screens/f'{n}-{width}.png'))
   for i in range(page.locator('figure').count()):
    figure=page.locator('figure').nth(i)
    figure.screenshot(path=str(screens/f'{n}-{width}-figure-{i}.png'))
    if width==375 and figure.evaluate('(x)=>x.scrollWidth>x.clientWidth+1'):
     figure.evaluate('(x)=>x.tabIndex=0');figure.focus();figure.press('End');figure.evaluate('(x)=>x.scrollLeft=x.scrollWidth')
     assert figure.evaluate('(x)=>x.scrollLeft>0')
     figure.screenshot(path=str(screens/f'{n}-{width}-figure-{i}-end.png'))
     figure.evaluate('(x)=>x.scrollLeft=0')
   rows.append(dict(lesson=n,width=width,status='PASS',figures=page.locator('figure img').count()))
  page.emulate_media(media='print')
  assert page.locator('figure img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
  page.emulate_media(media='screen')
  context=browser.new_context(java_script_enabled=False,viewport=dict(width=375,height=900))
  plain=context.new_page();resp=plain.goto(base+'/lessons/'+lesson.name,wait_until='networkidle');assert resp.status==200
  assert plain.locator('h1').first.is_visible() and not plain.evaluate('document.documentElement.scrollWidth>innerWidth+1')
  context.close()
  print('Browser PASS',n,flush=True)
 browser.close()
if server:server.shutdown();server.server_close()
report=dict(status='PASS' if not errors+http+failed else 'FAIL',base=base,views=rows,gallery_links=20,javascript_errors=errors,http_errors=http,failed_requests=failed,print_lessons=10,no_js_lessons=10,screenshots=str(screens))
(D/('live-browser.json' if live else 'browser.json')).write_text(json.dumps(report,indent=2)+'\n');print(report['status'],errors,http,failed);assert report['status']=='PASS'
